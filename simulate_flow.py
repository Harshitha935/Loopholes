import ast
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama

from main import (
    filter_chunks,
    generate_section,
    get_or_build_chunk_index,
    retrieve_top_k,
    retrieve_top_k_scored,
)
from utils import read_file_as_text


MAIN_DIR = Path("uploads/main")
SUPPORTING_DIR = Path("uploads/supporting")
EVENTS_FILE = Path("logs/tool_events.jsonl")


def append_event(run_id: str, tool: str, event: str, message: str, duration_ms: int = 0) -> None:
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "run_id": run_id,
                    "tool": tool,
                    "event": event,
                    "message": message,
                    "duration_ms": duration_ms,
                }
            )
            + "\n"
        )


def run_tool(run_id: str, tool: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    append_event(run_id, tool, "start", f"Calling {getattr(fn, '__name__', str(fn))}", 0)
    out = fn(*args, **kwargs)
    ms = int((time.perf_counter() - t0) * 1000)
    append_event(run_id, tool, "success", "Completed", ms)
    return out


def parse_inline_structure(text: str) -> dict:
    structure: dict = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        m = re.match(r"^(.+?)\s*\((.+)\)$", line)
        if m:
            parent = m.group(1).strip()
            children = [c.strip() for c in m.group(2).split(",") if c.strip()]
            structure[parent] = {child: {} for child in children}
        else:
            structure[line] = {}
    return structure


def flatten_structure_keys(structure: dict, parent: str = "") -> list[str]:
    out: list[str] = []
    for key, value in structure.items():
        path = f"{parent} > {key}" if parent else key
        out.append(path)
        if isinstance(value, dict) and value:
            out.extend(flatten_structure_keys(value, path))
    return out


def extract_dict_from_text(text: str) -> dict | None:
    text = (text or "").strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def normalize_structure(candidate) -> dict:
    if not isinstance(candidate, dict):
        return {}
    out: dict = {}
    for k, v in candidate.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = normalize_structure(v) if isinstance(v, dict) else {}
    return out


def propose_structure(chunks: list[dict], original: dict) -> dict:
    llm = ChatOllama(model="llama3.2:latest", temperature=0, num_predict=180)
    sample = "\n\n".join(
        f"[{c.get('source','')}|p{c.get('page','')}]\n{(c.get('text','') or '')[:250]}"
        for c in chunks[:8]
    )
    prompt = f"""
Build report structure from evidence only.
Current structure:
{json.dumps(original, indent=2)}

Evidence:
{sample}

Return strict JSON dict only with nested dict values.
"""
    txt = llm.invoke(prompt).content
    parsed = normalize_structure(extract_dict_from_text(txt) or {})
    return parsed or original


def main() -> None:
    main_files = [p for p in MAIN_DIR.iterdir() if p.is_file()]
    supporting = [str(p) for p in SUPPORTING_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not main_files or not supporting:
        raise RuntimeError("Need uploaded main/supporting documents first.")

    run_id = "sim-" + uuid.uuid4().hex[:8]

    structure_text = run_tool(run_id, "read_structure", read_file_as_text, str(main_files[0]))
    structure = parse_inline_structure(structure_text)

    chunks, embeddings, _ = run_tool(run_id, "read_supporting", get_or_build_chunk_index, supporting)

    t0 = time.perf_counter()
    append_event(run_id, "check_structure", "start", "Calling relevance score", 0)
    sections = flatten_structure_keys(structure)[:8]
    scores = []
    for sec in sections:
        scored = retrieve_top_k_scored(sec, chunks, embeddings, top_k=2)
        scores.append(max([s for _, s in scored[:2]] or [0.0]))
    rel = sum(scores) / max(1, len(scores))
    append_event(run_id, "check_structure", "success", "Completed", int((time.perf_counter() - t0) * 1000))
    append_event(run_id, "check_structure", "info", f"avg_relevance_score={rel:.3f}", 0)

    # Force a propose call in simulation to validate node/log path.
    final_structure = run_tool(run_id, "propose_structure", propose_structure, chunks, structure)

    report: dict[str, str] = {}
    for sec in flatten_structure_keys(final_structure)[:3]:
        retrieved = run_tool(run_id, "retrieve", retrieve_top_k, sec, chunks, embeddings, 3)
        selected = filter_chunks(retrieved, top_k=3)
        existing = "\n\n".join(report.values())
        report[sec] = run_tool(run_id, "generate", generate_section, sec, selected, existing)

    chat_llm = ChatOllama(model="llama3.2:latest", temperature=0.2, num_predict=120)
    ctx = "\n\n".join(f"{k}\n{v}" for k, v in report.items())
    run_tool(run_id, "chat", chat_llm.invoke, f"Answer naturally from this report:\n{ctx}\n\nUser: summarize")
    first_sec = next(iter(report.keys()))
    run_tool(run_id, "chat_edit", chat_llm.invoke, f"Edit section {first_sec} to be clearer:\n{report[first_sec]}")

    print(run_id)


if __name__ == "__main__":
    main()
