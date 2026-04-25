import re
import time
import json
import hashlib
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings


load_dotenv()

# Silence noisy PDF font-width parser logs from PyPDF2/pypdf.
for _logger_name in ("PyPDF2", "PyPDF2._cmap", "pypdf", "pypdf._cmap"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

FAST_MAX_PAGES_PER_DOC = 18
FAST_MAX_CHUNKS_PER_DOC = 140
CACHE_DIR = Path(".cache/rag")
ULTRA_TURBO_MODE = True
ULTRA_TURBO_SUBSECTIONS_EXTRACTIVE = False
ULTRA_TURBO_MAX_CONTEXT_CHARS = 2200

_llm = ChatOllama(model="llama3.2:latest", temperature=0, num_predict=120)
_embedding_model = None
_embedding_backend = "unknown"
_embedding_error: str | None = None


def _doc_index_fingerprint(document_paths: list[str]) -> str:
    parts: list[str] = []
    for p in sorted(document_paths):
        path = Path(p)
        try:
            st = path.stat()
            parts.append(f"{path.resolve()}|{int(st.st_mtime)}|{st.st_size}")
        except OSError:
            parts.append(f"{path}|missing")
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _cache_paths(fingerprint: str) -> tuple[Path, Path, Path]:
    base = CACHE_DIR / fingerprint
    return base / "chunks.json", base / "embeddings.npy", base / "meta.json"


def _load_cached_chunk_index(document_paths: list[str]) -> tuple[list[dict[str, Any]], np.ndarray] | None:
    fingerprint = _doc_index_fingerprint(document_paths)
    chunks_path, emb_path, meta_path = _cache_paths(fingerprint)
    if not (chunks_path.exists() and emb_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path)
        if not isinstance(chunks, list):
            return None
        if not isinstance(meta, dict):
            return None
        if int(meta.get("chunk_count", -1)) != len(chunks):
            return None
        if embeddings.shape[0] != len(chunks):
            return None
        return chunks, embeddings.astype(np.float32)
    except Exception:
        return None


def _save_cached_chunk_index(
    document_paths: list[str], chunks: list[dict[str, Any]], embeddings: np.ndarray
) -> None:
    fingerprint = _doc_index_fingerprint(document_paths)
    chunks_path, emb_path, meta_path = _cache_paths(fingerprint)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fingerprint": fingerprint,
        "chunk_count": len(chunks),
        "embedding_shape": list(embeddings.shape),
        "created_ts": time.time(),
    }
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(emb_path, embeddings.astype(np.float32))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def get_or_build_chunk_index(document_paths: list[str]) -> tuple[list[dict[str, Any]], np.ndarray, bool]:
    """
    Return (chunks, embeddings, cache_hit). Uses disk cache keyed by file fingerprint.
    """
    cached = _load_cached_chunk_index(document_paths)
    if cached is not None:
        chunks, embeddings = cached
        return chunks, embeddings, True
    chunks = ingest_pdfs(document_paths)
    embeddings = embed_chunk_records(chunks)
    _save_cached_chunk_index(document_paths, chunks, embeddings)
    return chunks, embeddings, False


def _get_embedding_model():
    """Lazy-load Ollama embeddings; keep app alive with fallback."""
    global _embedding_model, _embedding_backend, _embedding_error
    if _embedding_model is not None:
        return _embedding_model
    try:
        _embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        _embedding_backend = "ollama-embeddings"
        _embedding_error = None
    except Exception as exc:
        _embedding_model = None
        _embedding_backend = "hashing-fallback"
        _embedding_error = str(exc)
    return _embedding_model


def _encode_texts(texts: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    if model is not None:
        vectors = np.array(model.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    # Fallback: sklearn hashing vectors (no torch dependency).
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
    mat = vectorizer.transform(texts)
    return mat.toarray().astype(np.float32)


def get_embedding_backend_info() -> dict[str, str]:
    """
    Return active embedding backend metadata for UI/debug transparency.
    """
    _get_embedding_model()
    return {
        "backend": _embedding_backend,
        "error": _embedding_error or "",
    }


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < 4:
        return ""
    alpha_chars = sum(ch.isalpha() for ch in line)
    if alpha_chars < max(3, len(line) * 0.4):
        return ""
    if re.search(r"^\s*(page|www\.|http|copyright|figure|table)\b", line.lower()):
        return ""
    return line


def clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned = [_clean_line(ln) for ln in lines]
    cleaned = [ln for ln in cleaned if ln]
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_fast_excerpt(text: str) -> str:
    """
    Keep only salient snippets for speed:
    - first 3 lines
    - first and last sentence from each paragraph
    - last 3 lines
    """
    if not text.strip():
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    picked: list[str] = []

    def add_piece(piece: str) -> None:
        p = piece.strip()
        if p and p not in picked:
            picked.append(p)

    for ln in lines[:3]:
        add_piece(ln)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    for para in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if not sents:
            continue
        add_piece(sents[0])
        if len(sents) > 1:
            add_piece(sents[-1])

    for ln in lines[-3:]:
        add_piece(ln)

    excerpt = "\n\n".join(picked).strip()
    return excerpt if excerpt else text


def _is_low_text_density(page_text: str) -> bool:
    words = page_text.split()
    return len(words) < 80


def chunk_text(text: str, min_words: int = 300, max_words: int = 500) -> list[str]:
    """
    Paragraph-aware chunking in ~300-500 word windows.
    Preserves semantic boundaries as much as possible.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())
        if pw == 0:
            continue

        if current and current_words + pw > max_words:
            chunk = "\n\n".join(current).strip()
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
                current = []
                current_words = 0
            else:
                # Allow a bit larger chunk rather than producing tiny fragments.
                current.append(para)
                current_words += pw
                continue

        if pw > max_words:
            words = para.split()
            for i in range(0, len(words), max_words):
                part = " ".join(words[i : i + max_words]).strip()
                if part:
                    chunks.append(part)
            continue

        current.append(para)
        current_words += pw

    if current:
        chunks.append("\n\n".join(current).strip())

    merged: list[str] = []
    for ch in chunks:
        if merged and len(ch.split()) < min_words:
            merged[-1] = f"{merged[-1]}\n\n{ch}".strip()
        else:
            merged.append(ch)
    return merged


def ingest_pdfs(pdf_paths: list[str]) -> list[dict[str, Any]]:
    """
    Extract PDFs page-by-page using PyPDF2, clean noise, skip low-density pages,
    and return chunk records with metadata.
    """
    records: list[dict[str, Any]] = []
    for path_str in pdf_paths:
        path = Path(path_str)
        if path.suffix.lower() != ".pdf":
            continue

        reader = PdfReader(str(path))
        processed_pages = 0
        doc_chunk_count = 0
        for page_num, page in enumerate(reader.pages, start=1):
            if processed_pages >= FAST_MAX_PAGES_PER_DOC:
                break
            raw_text = page.extract_text() or ""
            if _is_low_text_density(raw_text):
                continue
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue
            fast_view = build_fast_excerpt(cleaned)
            if not fast_view:
                continue

            processed_pages += 1
            for ch in chunk_text(fast_view, min_words=70, max_words=160):
                records.append(
                    {
                        "text": ch,
                        "source": path.name,
                        "page": page_num,
                    }
                )
                doc_chunk_count += 1
                if doc_chunk_count >= FAST_MAX_CHUNKS_PER_DOC:
                    break
            if doc_chunk_count >= FAST_MAX_CHUNKS_PER_DOC:
                break
    return records


def embed_chunk_records(chunks: list[dict[str, Any]]) -> np.ndarray:
    """Encode all chunks once and keep embeddings in memory."""
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)
    texts = [c["text"] for c in chunks]
    return _encode_texts(texts)


def _split_section_key(section_key: str) -> tuple[str, str]:
    parts = [p.strip() for p in section_key.split(">", maxsplit=1)]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def retrieve_top_k(
    section_key: str, chunks: list[dict[str, Any]], chunk_embeddings: np.ndarray, top_k: int = 3
) -> list[dict[str, Any]]:
    """Cosine retrieval for one section/subsection query."""
    if not chunks or chunk_embeddings.size == 0:
        return []

    section, subsection = _split_section_key(section_key)
    query = f"{section} {subsection} supporting document evidence".strip()
    q_vec = _encode_texts([query])[0]
    scores = chunk_embeddings @ q_vec
    candidate_indices = np.argsort(scores)[::-1][: max(12, top_k * 3)]
    return [chunks[int(i)] for i in candidate_indices]


def retrieve_top_k_scored(
    section_key: str, chunks: list[dict[str, Any]], chunk_embeddings: np.ndarray, top_k: int = 3
) -> list[tuple[dict[str, Any], float]]:
    """
    Retrieve chunks with cosine scores for structure-relevance checks.
    """
    if not chunks or chunk_embeddings.size == 0:
        return []
    section, subsection = _split_section_key(section_key)
    query = f"{section} {subsection} supporting document evidence".strip()
    q_vec = _encode_texts([query])[0]
    scores = chunk_embeddings @ q_vec
    candidate_indices = np.argsort(scores)[::-1][: max(12, top_k * 3)]
    return [(chunks[int(i)], float(scores[int(i)])) for i in candidate_indices]


def _is_low_information(text: str) -> bool:
    words = text.split()
    if len(words) < 40:
        return True
    uniq = len(set(w.lower() for w in words))
    return (uniq / max(1, len(words))) < 0.18


def filter_chunks(chunks: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """
    Remove overlaps/low-info chunks and encourage source diversity.
    """
    filtered: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    for ch in chunks:
        text = re.sub(r"\s+", " ", ch["text"]).strip()
        if _is_low_information(text):
            continue

        duplicate = False
        for sel in filtered:
            a = text[:400].lower()
            b = sel["text"][:400].lower()
            if a == b:
                duplicate = True
                break
            if SequenceMatcher(None, a, b).ratio() > 0.86:
                duplicate = True
                break
        if duplicate:
            continue

        src = ch["source"]
        if source_counts.get(src, 0) >= 2:
            continue

        source_counts[src] = source_counts.get(src, 0) + 1
        filtered.append({**ch, "text": text})
        if len(filtered) >= top_k:
            break

    # Fill remaining slots if strict source cap left too few.
    if len(filtered) < top_k:
        for ch in chunks:
            if len(filtered) >= top_k:
                break
            if any(ch["text"][:120] == s["text"][:120] for s in filtered):
                continue
            if _is_low_information(ch["text"]):
                continue
            filtered.append(ch)

    return filtered[:top_k]


def _build_section_prompt(section_key: str, chunks_text: str, existing_context: str = "") -> str:
    return f"""
You are generating a report section.

Section: {section_key}

Instructions:
* Use the provided content as the PRIMARY source.
* Merge similar ideas into one concise explanation.
* Avoid repetition.
* Keep this section distinct from already written sections.
* If evidence is limited, lightly generalize while staying relevant.
* Do not invent unsupported facts.
* Write one coherent paragraph (no bullets).

Already written sections context (avoid repeating this):
{existing_context}

Content:
{chunks_text}
"""


def _build_fast_context(selected_chunks: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for c in selected_chunks[:2]:
        snippet = re.sub(r"\s+", " ", c["text"]).strip()
        if len(snippet) > 380:
            snippet = snippet[:380].rsplit(" ", 1)[0] + "..."
        pieces.append(f"[{c['source']} | page {c['page']}] {snippet}")
    text = " ".join(pieces).strip()
    return text[:ULTRA_TURBO_MAX_CONTEXT_CHARS]


def generate_section(
    section_key: str, selected_chunks: list[dict[str, Any]], existing_report_text: str = ""
) -> str:
    if not selected_chunks:
        return "No strong evidence was retrieved for this section from the provided PDFs."

    # In ultra-turbo mode, keep subsection generation extractive for speed.
    if ULTRA_TURBO_MODE and ULTRA_TURBO_SUBSECTIONS_EXTRACTIVE and ">" in section_key:
        return _fallback_text(selected_chunks)

    if ULTRA_TURBO_MODE:
        fast_context = _build_fast_context(selected_chunks)
        if fast_context:
            prompt = _build_section_prompt(section_key, fast_context, existing_report_text[:1200])
            response = _llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            content = content.strip()
            if content:
                return content
        return _fallback_text(selected_chunks)

    context = "\n\n".join(
        f"[{c['source']} | page {c['page']}]\n{c['text']}" for c in selected_chunks
    )
    prompt = _build_section_prompt(section_key, context, existing_report_text[:1600])
    response = _llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    content = content.strip()
    if not content:
        return "No strong evidence was retrieved for this section from the provided PDFs."
    return content


def _fallback_text(selected_chunks: list[dict[str, Any]]) -> str:
    if not selected_chunks:
        return "No strong evidence was retrieved for this section from the provided PDFs."
    text = " ".join(c["text"] for c in selected_chunks)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return " ".join(sents[:3]) if sents else selected_chunks[0]["text"][:500]


def evaluate_structure_relevance(
    structure: dict[str, Any], report: dict[str, str]
) -> dict[str, str]:
    """
    Check if headings are relevant to generated content.
    Returns dict with:
      - status: "ok" or "suggest_change"
      - message: short explanation
      - suggested_structure: optional proposed structure text
    """
    section_list = "\n".join(report.keys())
    report_text = "\n\n".join(f"{k}\n{v}" for k, v in report.items())
    prompt = f"""
You are validating whether report headings match report content.

Current structure headings:
{section_list}

Generated report:
{report_text[:7000]}

Respond in exactly this format:
STATUS: <ok or suggest_change>
MESSAGE: <one short sentence>
SUGGESTED_STRUCTURE:
<if suggest_change: provide improved headings, one per line; else write NONE>
"""
    response = _llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    status = "ok"
    message = "Structure appears relevant to the generated content."
    suggested = "NONE"

    status_match = re.search(r"STATUS:\s*(.+)", content, flags=re.IGNORECASE)
    if status_match:
        parsed = status_match.group(1).strip().lower()
        if "suggest" in parsed:
            status = "suggest_change"
    message_match = re.search(r"MESSAGE:\s*(.+)", content, flags=re.IGNORECASE)
    if message_match:
        message = message_match.group(1).strip()
    suggested_match = re.search(
        r"SUGGESTED_STRUCTURE:\s*(.+)$", content, flags=re.IGNORECASE | re.DOTALL
    )
    if suggested_match:
        suggested = suggested_match.group(1).strip()

    return {
        "status": status,
        "message": message,
        "suggested_structure": suggested,
    }


def _flatten_sections(structure: dict[str, Any], parent: str = "") -> list[str]:
    """Preserve existing structure traversal logic."""
    sections: list[str] = []
    for key, value in structure.items():
        name = f"{parent} > {key}" if parent else key
        sections.append(name)
        if isinstance(value, dict) and value:
            sections.extend(_flatten_sections(value, name))
    return sections


def _log_selected_chunks(section_key: str, selected_chunks: list[dict[str, Any]]) -> None:
    print(f"\nSection: {section_key}")
    print("Retrieved:")
    for chunk in selected_chunks:
        preview = re.sub(r"\s+", " ", chunk["text"])[:100]
        print(f"- {chunk['source']} | page {chunk['page']} | {preview}")


def run_pipeline(
    structure: dict[str, Any],
    document_paths: list[str],
    top_k: int = 3,
    time_budget_seconds: int = 20,
) -> dict[str, str]:
    """
    Structure-guided RAG over uploaded supporting documents.
    Output format:
    {
      "Section": "text...",
      "Section > Subsection": "text..."
    }
    """
    start = time.perf_counter()
    _get_embedding_model()
    if _embedding_error:
        print(
            f"[warning] embedding backend fallback in use ({_embedding_backend}): {_embedding_error}"
        )

    chunks, chunk_embeddings, cache_hit = get_or_build_chunk_index(document_paths)
    if cache_hit:
        print("[cache] Reused cached chunks+embeddings index")
    sections = _flatten_sections(structure)
    if not chunks:
        return {
            section_key: "No strong evidence was retrieved for this section from the provided PDFs."
            for section_key in sections
        }
    final_report: dict[str, str] = {}

    for section_key in sections:
        elapsed = time.perf_counter() - start
        retrieved = retrieve_top_k(section_key, chunks, chunk_embeddings, top_k=top_k)
        selected = filter_chunks(retrieved, top_k=top_k)
        _log_selected_chunks(section_key, selected)
        if elapsed > time_budget_seconds:
            final_report[section_key] = _fallback_text(selected)
        else:
            prior_text = "\n\n".join(v for k, v in final_report.items() if k != section_key and v)
            final_report[section_key] = generate_section(section_key, selected, prior_text)

    total_elapsed = time.perf_counter() - start
    print(f"\n[done] Generated {len(final_report)} sections in {total_elapsed:.2f}s")
    return final_report


if __name__ == "__main__":
    # Minimal example usage:
    structure = {
        "Introduction": {},
        "Methodology": {"Data Collection": {}, "Analysis": {}},
        "Findings": {},
        "Conclusion": {},
    }
    docs = [
        "uploads/supporting/sustainability-07-07833.pdf",
        "uploads/supporting/Sustainable-Agriculture.pdf",
    ]

    report = run_pipeline(structure, docs)
    print("\nFinal report dict:")
    print(report)