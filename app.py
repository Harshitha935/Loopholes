import ast
import json
import os
import re
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# Work around Streamlit watcher crash with torch custom classes on Windows.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import streamlit.components.v1 as components
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_ollama import ChatOllama

from utils import read_file_as_text

MAIN_DIR = Path("uploads/main")
SUPPORTING_DIR = Path("uploads/supporting")
LOG_DIR = Path("logs")
EVENTS_FILE = LOG_DIR / "tool_events.jsonl"
RUN_LOG_DIR = LOG_DIR / "runs"
RUN_STREAM_DIR = LOG_DIR / "streams"
ALLOWED_TYPES = ["txt", "md", "pdf", "docx"]

TRACKED_TOOLS = [
    "read_structure",
    "read_supporting",
    "check_structure",
    "propose_structure",
    "retrieve",
    "generate",
    "chat",
    "chat_edit",
    "chat_read",
    "chat_retrieve",
    "chat_generate",
    "chat_read_supporting_docs",
    "chat_rag_pipeline",
]

CHAT_LLM = ChatOllama(model="llama3.2:latest", temperature=0.2, num_predict=180)
STRUCTURE_LLM = ChatOllama(model="llama3.2:latest", temperature=0, num_predict=220)
_AGENT = None


def _init_state() -> None:
    st.session_state.setdefault("page", "upload")
    st.session_state.setdefault("structure", {})
    st.session_state.setdefault("report", {})
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("supporting_paths", [])
    st.session_state.setdefault("run_id", uuid.uuid4().hex[:12])
    st.session_state.setdefault("dev_window_opened", False)
    st.session_state.setdefault("structure_origin", "original")
    st.session_state.setdefault("chat_locked", False)
    st.session_state.setdefault("_step_counter", 0)
    st.session_state.setdefault("_run_trace", {"run_id": st.session_state["run_id"], "steps": [], "user_feedback": None})


def _reset_run_trace(run_id: str) -> None:
    st.session_state["run_id"] = run_id
    st.session_state["_step_counter"] = 0
    st.session_state["_run_trace"] = {"run_id": run_id, "steps": [], "user_feedback": None}
    _persist_run_trace()


def _safe_data(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _short_preview(value, max_len: int = 240) -> str:
    txt = value if isinstance(value, str) else json.dumps(_safe_data(value), ensure_ascii=True, default=str)
    txt = re.sub(r"\s+", " ", txt).strip()
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3] + "..."


def _persist_run_trace() -> None:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = st.session_state.get("run_id", "")
    payload = st.session_state.get("_run_trace", {"run_id": run_id, "steps": [], "user_feedback": None})
    path = RUN_LOG_DIR / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, default=str, indent=2), encoding="utf-8")


def _append_run_step(step: dict) -> None:
    trace = st.session_state.get("_run_trace")
    if not isinstance(trace, dict):
        return
    steps = trace.setdefault("steps", [])
    if isinstance(steps, list):
        steps.append(step)
        _persist_run_trace()


def _update_run_step(step_id: str, updates: dict) -> None:
    trace = st.session_state.get("_run_trace")
    if not isinstance(trace, dict):
        return
    steps = trace.get("steps", [])
    if not isinstance(steps, list):
        return
    for i in range(len(steps) - 1, -1, -1):
        row = steps[i]
        if isinstance(row, dict) and row.get("step_id") == step_id:
            row.update(updates)
            _persist_run_trace()
            return


def _stream_log_path(run_id: str) -> Path:
    RUN_STREAM_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_STREAM_DIR / f"{run_id}.log"


def _append_stream_line(run_id: str, line: str) -> None:
    if not run_id:
        return
    path = _stream_log_path(run_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _read_stream_lines(run_id: str, max_lines: int = 120) -> list[str]:
    if not run_id:
        return []
    path = _stream_log_path(run_id)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _load_run_trace(run_id: str) -> dict:
    if not run_id:
        return {"run_id": "", "steps": [], "user_feedback": None}
    path = RUN_LOG_DIR / f"{run_id}.json"
    if not path.exists():
        return {"run_id": run_id, "steps": [], "user_feedback": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"run_id": run_id, "steps": [], "user_feedback": None}


def _list_recent_run_ids(limit: int = 20) -> list[str]:
    if not RUN_LOG_DIR.exists():
        return []
    files = sorted(
        RUN_LOG_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    run_ids: list[str] = []
    for p in files[: max(1, limit)]:
        rid = p.stem.strip()
        if rid:
            run_ids.append(rid)
    return run_ids


def _accumulated_tool_scores(run_ids: list[str]) -> list[dict]:
    """
    Aggregate tool scores across multiple runs for a single summary table.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    fixing_counts: dict[str, int] = {}

    for rid in run_ids:
        trace = _load_run_trace(rid)
        analysis = analyze_run(trace, feedback_text="", previous_runs=[])
        summary = analysis.get("summary", {}) if isinstance(analysis, dict) else {}
        rows = summary.get("tool_table", []) if isinstance(summary, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tool = str(row.get("tool", "")).strip()
            if not tool:
                continue
            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0
            totals[tool] = totals.get(tool, 0.0) + score
            counts[tool] = counts.get(tool, 0) + 1
            if str(row.get("needs_fixing", "")).upper() == "YES":
                fixing_counts[tool] = fixing_counts.get(tool, 0) + 1

    table: list[dict] = []
    for tool, total in totals.items():
        c = max(1, counts.get(tool, 0))
        table.append(
            {
                "tool": tool,
                "total_score": f"{total:.2f}",
                "avg_score": f"{(total / c):.2f}",
                "runs_seen": c,
                "needs_fixing_runs": fixing_counts.get(tool, 0),
            }
        )

    return sorted(table, key=lambda r: float(r["total_score"]), reverse=True)


def build_graph(run_logs: dict | list[dict]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    Build a DAG view from one run's step logs.

    Returns:
      - nodes: {step_id: tool_name}
      - edges: [(dependency_step_id, step_id), ...]

    Accepted input shapes:
      - {"run_id": "...", "steps": [...]}
      - [{"step_id": "...", ...}, ...]
    """
    if isinstance(run_logs, dict):
        steps = run_logs.get("steps", []) or []
    elif isinstance(run_logs, list):
        steps = run_logs
    else:
        return {}, []

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id", "")).strip()
        tool_name = str(step.get("tool_name", "")).strip()
        if not step_id:
            continue
        nodes[step_id] = tool_name
        deps = step.get("dependencies", []) or []
        if isinstance(deps, list):
            for dep in deps:
                dep_id = str(dep).strip()
                if dep_id:
                    edges.append((dep_id, step_id))

    return nodes, edges


def parse_feedback(text: str) -> dict:
    """
    Rule-based feedback parser.

    Returns:
      {
        "targets": {target_name: weight},
        "severity": float
      }
    """
    if not isinstance(text, str) or not text.strip():
        return {"targets": {}, "severity": 0.3}

    low = text.lower()

    target_rules = {
        "structure": ["structure", "outline", "heading", "headings", "section", "flow"],
        "retrieval": ["retrieval", "retrieve", "rag", "source", "evidence", "chunk", "supporting"],
        "content": ["content", "generate", "generation", "response", "answer", "output", "write", "writing"],
    }

    targets: dict[str, float] = {}
    for name, kws in target_rules.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > 0:
            # Keep simple bounded weights in [0.5, 1.0]
            targets[name] = min(1.0, 0.5 + 0.15 * hits)

    # Severity mapping: simple phrase-to-score rules.
    severity = 0.5
    severity_rules = [
        (0.9, ["sucks", "terrible", "awful", "horrible", "broken", "worst"]),
        (0.8, ["very bad", "really bad", "garbage", "useless"]),
        (0.7, ["bad", "wrong", "poor"]),
        (0.6, ["not good", "issue", "problem"]),
        (0.4, ["slightly bad", "a bit bad", "somewhat off"]),
        (0.3, ["slightly", "minor", "little", "small issue"]),
    ]
    for sev, phrases in severity_rules:
        if any(p in low for p in phrases):
            severity = sev
            break

    # Clamp to requested range.
    severity = max(0.3, min(0.9, float(severity)))
    return {"targets": targets, "severity": severity}


def _apply_feedback_to_blame_scores(
    dag_scores: dict[str, float], feedback: dict
) -> dict[str, float]:
    """
    final_score = DAG_score * feedback_weight
    where mentioned tools/categories are boosted by severity.
    """
    if not dag_scores:
        return {}

    targets = feedback.get("targets", {}) if isinstance(feedback, dict) else {}
    severity = float(feedback.get("severity", 0.3)) if isinstance(feedback, dict) else 0.3
    severity = max(0.3, min(0.9, severity))

    if not targets:
        return dag_scores

    # Map actual tool names to coarse categories parsed from feedback.
    tool_to_category = {
        "read_structure": "structure",
        "check_structure": "structure",
        "propose_structure": "structure",
        "read_supporting": "retrieval",
        "retrieve": "retrieval",
        "chat_retrieve": "retrieval",
        "chat_read_supporting_docs": "retrieval",
        "rag_pipeline": "retrieval",
        "chat_rag_pipeline": "retrieval",
        "generate": "content",
        "chat_generate": "content",
        "chat_edit": "content",
        "chat": "content",
        "chat_read": "content",
    }

    boosted: dict[str, float] = {}
    for tool, base_score in dag_scores.items():
        category = tool_to_category.get(tool, "")
        target_weight = float(targets.get(category, 0.0))
        # 1.0 means no change; mention + severity boosts score.
        feedback_weight = 1.0 + (severity * target_weight)
        boosted[tool] = base_score * feedback_weight

    total = sum(boosted.values())
    if total <= 0:
        return {}
    return {tool: score / total for tool, score in boosted.items()}


def compute_blame_scores(run: dict, feedback_text: str | None = None) -> dict[str, float]:
    """
    Compute normalized tool-level blame attribution for a run DAG.

    Logic:
      1) Start at final step with blame 1.0
      2) Propagate backward through dependencies
      3) Split blame equally among dependencies
      4) Accumulate blame per step
      5) Aggregate by tool_name and normalize to sum to 1.0
    """
    if not isinstance(run, dict):
        return {}

    steps = run.get("steps", []) or []
    if not isinstance(steps, list) or not steps:
        return {}

    # Optional gate: only attribute blame for negative feedback runs.
    feedback = run.get("user_feedback")
    if isinstance(feedback, (int, float)) and feedback >= 0:
        return {}

    ordered_ids: list[str] = []
    step_by_id: dict[str, dict] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("step_id", "")).strip()
        if not sid:
            continue
        ordered_ids.append(sid)
        step_by_id[sid] = step

    if not ordered_ids:
        return {}

    final_step_id = ordered_ids[-1]
    step_blame: dict[str, float] = {}
    stack: list[tuple[str, float]] = [(final_step_id, 1.0)]

    while stack:
        sid, blame = stack.pop()
        if sid not in step_by_id or blame <= 0:
            continue
        step_blame[sid] = step_blame.get(sid, 0.0) + blame
        deps = step_by_id[sid].get("dependencies", []) or []
        dep_ids = [str(d).strip() for d in deps if str(d).strip()]
        if not dep_ids:
            continue
        share = blame / len(dep_ids)
        for dep_id in dep_ids:
            stack.append((dep_id, share))

    tool_scores: dict[str, float] = {}
    for sid, blame in step_blame.items():
        tool = str(step_by_id.get(sid, {}).get("tool_name", "")).strip()
        if not tool:
            continue
        tool_scores[tool] = tool_scores.get(tool, 0.0) + blame

    total = sum(tool_scores.values())
    if total <= 0:
        return {}
    dag_scores = {tool: score / total for tool, score in tool_scores.items()}
    if feedback_text:
        feedback = parse_feedback(feedback_text)
        return _apply_feedback_to_blame_scores(dag_scores, feedback)
    return dag_scores


def analyze_tools(tool_scores: dict[str, float], steps: list[dict]) -> list[dict]:
    """
    Convert tool attribution scores into deterministic actionable analysis.

    Returns:
      [
        {
          "tool_name": "...",
          "score": float,
          "needs_fixing": bool,
          "recommended_action": "..."
        },
        ...
      ]
    """
    if not isinstance(tool_scores, dict):
        return []

    steps = steps if isinstance(steps, list) else []

    tool_latency: dict[str, list[float]] = {}
    tool_errors: dict[str, int] = {}
    step_to_tool: dict[str, str] = {}
    upstream_refs: dict[str, int] = {}

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id", "")).strip()
        tool = str(step.get("tool_name", "")).strip()
        if not tool:
            continue
        if step_id:
            step_to_tool[step_id] = tool

        latency = float(step.get("latency_ms", 0.0) or 0.0)
        tool_latency.setdefault(tool, []).append(latency)

        status = str(step.get("status", "")).strip().lower()
        if status == "error":
            tool_errors[tool] = tool_errors.get(tool, 0) + 1

    # Count how often a tool is used as an upstream dependency by other steps.
    for step in steps:
        if not isinstance(step, dict):
            continue
        deps = step.get("dependencies", []) or []
        if not isinstance(deps, list):
            continue
        for dep in deps:
            dep_id = str(dep).strip()
            dep_tool = step_to_tool.get(dep_id, "")
            if dep_tool:
                upstream_refs[dep_tool] = upstream_refs.get(dep_tool, 0) + 1

    # Global latency baseline for simple "high latency" decision.
    all_latencies = [lat for vals in tool_latency.values() for lat in vals]
    avg_latency_global = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0
    latency_threshold = max(1000.0, avg_latency_global * 1.5)

    results: list[dict] = []
    for tool, score in sorted(tool_scores.items(), key=lambda x: x[1], reverse=True):
        score_f = float(score)
        needs_fixing = score_f >= 0.5

        latencies = tool_latency.get(tool, [])
        avg_latency_tool = (sum(latencies) / len(latencies)) if latencies else 0.0
        has_error = tool_errors.get(tool, 0) > 0
        is_upstream_heavy = upstream_refs.get(tool, 0) >= 2

        reasons: list[str] = []
        if needs_fixing:
            reasons.append("Review logic or prompt quality")
        if avg_latency_tool >= latency_threshold and latencies:
            reasons.append("Optimize performance")
        if is_upstream_heavy:
            reasons.append("Check input dependencies")
        if has_error:
            reasons.append("Improve robustness")

        recommended_action = "; ".join(reasons) if reasons else "Monitor behavior"
        results.append(
            {
                "tool_name": tool,
                "score": score_f,
                "needs_fixing": needs_fixing,
                "recommended_action": recommended_action,
            }
        )

    return results


def detect_frustration(messages) -> dict[str, object]:
    """
    Detect developer frustration from chat messages using keyword rules only.

    Returns:
      {
        "level": "low" | "medium" | "high",
        "repeat": bool
      }
    """
    texts: list[str] = []
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                txt = msg.get("content", "")
                if isinstance(txt, str):
                    texts.append(txt.lower())
            elif isinstance(msg, str):
                texts.append(msg.lower())
    elif isinstance(messages, str):
        texts.append(messages.lower())

    combined = " ".join(texts)

    # Repeat detection.
    repeat_tokens = ["no", "again", "still wrong"]
    repeat = any(tok in combined for tok in repeat_tokens)

    # Frustration level detection.
    high_tokens = ["sucks", "terrible", "nooooo"]
    medium_tokens = ["not working"]

    if any(tok in combined for tok in high_tokens):
        level = "high"
    elif any(tok in combined for tok in medium_tokens):
        level = "medium"
    else:
        level = "low"

    return {"level": level, "repeat": repeat}


def escalate_across_runs(
    current_run_analysis,
    previous_runs: list[dict[str, float]],
    frustration,
    lookback_n: int = 5,
) -> dict[str, object]:
    """
    Escalate attribution across multiple runs when frustration is high/repeated.

    Inputs:
      - current_run_analysis: current tool scores (dict) or analyze_tools output (list[dict])
      - previous_runs: list of past tool score dicts
      - frustration: {"level": "...", "repeat": bool} (or compatible dict)
      - lookback_n: max historical runs to include (default 5)

    Returns:
      {
        "root_issue": str,
        "evidence": list[str],
        "suggestion": str
      }
    """
    # Gate escalation.
    level = ""
    repeat = False
    if isinstance(frustration, dict):
        level = str(frustration.get("level", "")).lower().strip()
        repeat = bool(frustration.get("repeat", False))
    should_escalate = (level == "high") or repeat
    if not should_escalate:
        return {
            "root_issue": "No escalation needed",
            "evidence": ["Frustration signal is not high/repeated."],
            "suggestion": "Continue monitoring current run behavior.",
        }

    def to_score_map(obj) -> dict[str, float]:
        if isinstance(obj, dict):
            out: dict[str, float] = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out
        if isinstance(obj, list):
            out: dict[str, float] = {}
            for row in obj:
                if not isinstance(row, dict):
                    continue
                tool = str(row.get("tool_name", "")).strip()
                if not tool:
                    continue
                try:
                    out[tool] = float(row.get("score", 0.0))
                except Exception:
                    continue
            return out
        return {}

    current_scores = to_score_map(current_run_analysis)
    recent_prev = previous_runs[-max(0, int(lookback_n) - 1) :] if isinstance(previous_runs, list) else []
    score_series: list[dict[str, float]] = [r for r in [to_score_map(p) for p in recent_prev] if r]
    if current_scores:
        score_series.append(current_scores)

    if not score_series:
        return {
            "root_issue": "Insufficient data",
            "evidence": ["No tool scores found across current/previous runs."],
            "suggestion": "Collect at least one complete run with tool scores.",
        }

    # Aggregate averages and consistency counts.
    avg_scores: dict[str, float] = {}
    high_counts: dict[str, int] = {}
    run_count = len(score_series)
    for run_scores in score_series:
        for tool, score in run_scores.items():
            avg_scores[tool] = avg_scores.get(tool, 0.0) + float(score)
            if float(score) >= 0.4:
                high_counts[tool] = high_counts.get(tool, 0) + 1
    for tool in list(avg_scores.keys()):
        avg_scores[tool] = avg_scores[tool] / run_count

    # Consistently high = high in >= 60% runs and average >= 0.35
    consistent_tools = [
        tool
        for tool, avg in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        if (high_counts.get(tool, 0) / run_count) >= 0.6 and avg >= 0.35
    ]

    if consistent_tools:
        lead = consistent_tools[0]
        evidence = [
            f"{t}: avg_score={avg_scores[t]:.3f}, high_runs={high_counts.get(t,0)}/{run_count}"
            for t in consistent_tools[:3]
        ]
        return {
            "root_issue": f"It appears {lead} is consistently responsible across runs",
            "evidence": evidence,
            "suggestion": "Prioritize fixes for this tool's logic/prompt/dependencies before tuning downstream steps.",
        }

    # Fallback: point to highest average.
    top_tool = max(avg_scores.items(), key=lambda x: x[1])[0]
    return {
        "root_issue": f"No single consistently high tool; {top_tool} is currently highest on average",
        "evidence": [f"{k}: avg_score={v:.3f}" for k, v in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]],
        "suggestion": "Investigate top tool first, then validate cross-tool interactions in recent runs.",
    }


def analyze_run(run_logs: dict, feedback_text: str, previous_runs: list[dict[str, float]]) -> dict:
    """
    Build developer-visibility data payload (no UI).
    """
    # 1) Build DAG
    nodes, edges = build_graph(run_logs)

    # Normalize steps input from run logs.
    steps = []
    if isinstance(run_logs, dict):
        raw_steps = run_logs.get("steps", []) or []
        if isinstance(raw_steps, list):
            steps = raw_steps

    # 2) Compute DAG blame (without feedback weighting)
    dag_scores = compute_blame_scores(run_logs, feedback_text=None)

    # 3) Apply feedback weighting
    feedback = parse_feedback(feedback_text or "")
    weighted_scores = _apply_feedback_to_blame_scores(dag_scores, feedback) if dag_scores else {}

    # 4) Generate tool analysis
    tool_analysis = analyze_tools(weighted_scores, steps)
    sorted_analysis = sorted(tool_analysis, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    primary_issue = sorted_analysis[0]["tool_name"] if sorted_analysis else "unknown"

    tool_table = [
        {
            "tool": str(row.get("tool_name", "")),
            "score": f"{float(row.get('score', 0.0)):.2f}",
            "needs_fixing": "YES" if bool(row.get("needs_fixing", False)) else "NO",
            "action": str(row.get("recommended_action", "")),
        }
        for row in sorted_analysis
        if isinstance(row, dict)
    ]

    # Pipeline tools in execution order (deduplicated, first occurrence kept).
    pipeline: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool_name", "")).strip()
        if tool and tool not in pipeline:
            pipeline.append(tool)

    # Keep these arguments/values intentionally touched for deterministic analysis flow.
    _ = (previous_runs, nodes, edges)

    return {
        "summary": {
            "pipeline": pipeline,
            "primary_issue": primary_issue,
            "tool_table": tool_table,
        },
        "logs": {
            "collapsed": True,
            "steps": steps,
        },
    }


def _append_event(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")


def _run_tool(tool: str, fn, *, dependencies: list[str], input_data=None, **call_kwargs):
    if dependencies is None:
        raise ValueError("dependencies must be passed explicitly")
    st.session_state["_step_counter"] = int(st.session_state.get("_step_counter", 0)) + 1
    step_id = f"s{st.session_state['_step_counter']}"
    start = time.perf_counter()
    ts_start = time.time()
    start_ts = datetime.now().strftime("%H:%M:%S")
    _append_event(
        {
            "time": start_ts,
            "run_id": st.session_state["run_id"],
            "tool": tool,
            "event": "start",
            "message": f"Calling {getattr(fn, '__name__', str(fn))}",
            "duration_ms": 0,
        }
    )
    _append_run_step(
        {
            "step_id": step_id,
            "tool_name": tool,
            "timestamp_start": ts_start,
            "timestamp_end": None,
            "input": _safe_data(input_data if input_data is not None else call_kwargs),
            "output": None,
            "dependencies": dependencies,
            "status": "running",
            "latency_ms": 0.0,
        }
    )
    run_id = st.session_state["run_id"]
    start_line = (
        f"[RUN {run_id}] {step_id} {tool} start | deps={dependencies} | "
        f"input={_short_preview(input_data if input_data is not None else call_kwargs)}"
    )
    print(start_line)
    _append_stream_line(run_id, start_line)
    try:
        out = fn(**call_kwargs)
        ts_end = time.time()
        elapsed = int((time.perf_counter() - start) * 1000)
        _update_run_step(
            step_id,
            {
                "timestamp_end": ts_end,
                "output": _safe_data(out),
                "status": "success",
                "latency_ms": (ts_end - ts_start) * 1000.0,
            },
        )
        _append_event(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "run_id": st.session_state["run_id"],
                "tool": tool,
                "event": "success",
                "message": f"{step_id} completed | output={_short_preview(out)}",
                "duration_ms": elapsed,
            }
        )
        success_line = (
            f"[RUN {run_id}] {step_id} {tool} success | {elapsed} ms | "
            f"output={_short_preview(out)}"
        )
        print(success_line)
        _append_stream_line(run_id, success_line)
        return out, step_id
    except Exception as exc:
        ts_end = time.time()
        elapsed = int((time.perf_counter() - start) * 1000)
        _update_run_step(
            step_id,
            {
                "timestamp_end": ts_end,
                "output": _safe_data(str(exc)),
                "status": "error",
                "latency_ms": (ts_end - ts_start) * 1000.0,
            },
        )
        _append_event(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "run_id": st.session_state["run_id"],
                "tool": tool,
                "event": "error",
                "message": str(exc),
                "duration_ms": elapsed,
            }
        )
        error_line = (
            f"[RUN {run_id}] {step_id} {tool} error | {elapsed} ms | "
            f"error={_short_preview(str(exc))}"
        )
        print(error_line)
        _append_stream_line(run_id, error_line)
        raise


def _read_all_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    all_rows: list[dict] = []
    with EVENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_rows.append(obj)
    return all_rows


def _load_events(run_id: str) -> list[dict]:
    all_rows = _read_all_events()
    if not all_rows:
        return []
    if not run_id:
        latest_run = all_rows[-1].get("run_id", "")
        return [r for r in all_rows if r.get("run_id") == latest_run]
    rows = [r for r in all_rows if r.get("run_id") == run_id]
    if rows:
        return rows
    latest_run = all_rows[-1].get("run_id", "")
    return [r for r in all_rows if r.get("run_id") == latest_run]


def _resolve_live_run_id(all_events: list[dict], preferred_rid: str = "") -> str:
    """
    Resolve the run id that should be shown in live mode.
    Priority:
      1) preferred rid (if it is currently active),
      2) most recently active run (has unmatched start events),
      3) preferred rid (if it exists in logs),
      4) latest run in logs.
    """
    if not all_events:
        return preferred_rid

    stats: dict[str, dict[str, int]] = {}
    for idx, e in enumerate(all_events):
        rid = str(e.get("run_id", "")).strip()
        if not rid:
            continue
        ev = str(e.get("event", "")).strip()
        row = stats.setdefault(rid, {"start": 0, "end": 0, "last_idx": -1})
        if ev == "start":
            row["start"] += 1
        elif ev in {"success", "error"}:
            row["end"] += 1
        row["last_idx"] = idx

    if preferred_rid and preferred_rid in stats:
        row = stats[preferred_rid]
        if row["start"] > row["end"]:
            return preferred_rid

    active = [(rid, row["last_idx"]) for rid, row in stats.items() if row["start"] > row["end"]]
    if active:
        active.sort(key=lambda x: x[1], reverse=True)
        return active[0][0]

    if preferred_rid and preferred_rid in stats:
        return preferred_rid

    return str(all_events[-1].get("run_id", "")).strip()


def _tool_blocks(events: list[dict], steps: list[dict]) -> list[str]:
    seen: list[str] = []
    for t in TRACKED_TOOLS:
        if t not in seen:
            seen.append(t)
    for e in events:
        t = str(e.get("tool", "")).strip()
        if t and t not in seen:
            seen.append(t)
    for s in steps:
        t = str(s.get("tool_name", "")).strip()
        if t and t not in seen:
            seen.append(t)
    return seen


def _tool_status_map(events: list[dict], steps: list[dict], blocks: list[str]) -> dict[str, str]:
    status = {t: "idle" for t in blocks}
    for e in events:
        tool = e.get("tool")
        if tool not in status:
            continue
        ev = e.get("event")
        if ev == "start":
            status[tool] = "running"
        elif ev == "success":
            status[tool] = "success"
        elif ev == "error":
            status[tool] = "error"
        elif ev == "skipped":
            status[tool] = "skipped"

    # Ensure tools with only step logs (or missing events) still show status.
    for s in steps:
        tool = s.get("tool_name")
        if tool not in status:
            continue
        if status.get(tool) == "running":
            continue
        sts = s.get("status")
        if sts == "error":
            status[tool] = "error"
        elif sts == "success":
            status[tool] = "success"
    return status


def _status_visual(state: str) -> tuple[str, str]:
    if state == "running":
        return "#facc15", "running"
    if state == "success":
        return "#22c55e", "success"
    if state == "error":
        return "#ef4444", "error"
    if state == "skipped":
        return "#60a5fa", "skipped"
    return "#9ca3af", ""


def _render_flow_chart(status: dict[str, str], blocks: list[str]) -> None:
    parts = []
    for idx, tool in enumerate(blocks):
        color, css = _status_visual(status.get(tool, "idle"))
        parts.append(f"<div class='flow-block {css}' style='border-color:{color}; color:{color};'>{tool}</div>")
        if idx < len(blocks) - 1:
            parts.append("<div class='flow-arrow'>→</div>")
    html = f"""
    <style>
      .flow-wrap {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
      .flow-block {{
        border:1.8px solid #9ca3af;
        border-radius:12px;
        padding:9px 12px;
        min-width:120px;
        text-align:center;
        font-size:12px;
        font-weight:700;
        letter-spacing:0.2px;
        background:rgba(15,23,42,0.35);
      }}
      .flow-arrow {{ color:#64748b; font-weight:700; }}
      .running {{ animation:pulse 1s infinite; background:rgba(250,204,21,0.14); box-shadow:0 0 14px rgba(250,204,21,0.7); }}
      .success {{ background:rgba(34,197,94,0.12); box-shadow:0 0 10px rgba(34,197,94,0.45); }}
      .error {{ background:rgba(239,68,68,0.14); box-shadow:0 0 12px rgba(239,68,68,0.65); }}
      .skipped {{ background:rgba(96,165,250,0.12); box-shadow:0 0 10px rgba(96,165,250,0.5); }}
      @keyframes pulse {{ 0% {{transform:scale(1)}} 50% {{transform:scale(1.03)}} 100% {{transform:scale(1)}} }}
    </style>
    <div class='flow-wrap'>{''.join(parts)}</div>
    """
    st.markdown(html, unsafe_allow_html=True)


@st.fragment(run_every="900ms")
def _render_dev_live_fragment(rid: str) -> None:
    rid = st.query_params.get("rid", "")
    all_events = _read_all_events()
    # Live mode: prefer active run (unmatched start events), and keep rid if still active.
    target_run = _resolve_live_run_id(all_events, preferred_rid=rid)
    events = _load_events(target_run)
    shown_run = events[-1].get("run_id", target_run) if events else target_run
    trace = _load_run_trace(shown_run)
    steps = trace.get("steps", [])
    blocks = _tool_blocks(events, steps)
    status = _tool_status_map(events, steps, blocks)
    st.title("Developer Observability")
    m1, m2, m3 = st.columns(3)
    m1.metric("Run", shown_run[:12] if shown_run else "-")
    m2.metric("Events", str(len(events)))
    m3.metric("Steps", str(len(steps)))
    st.caption("Mode: live (currently executing run)")

    st.markdown("### Execution Flow")
    with st.container(border=True):
        _render_flow_chart(status, blocks)
    stream_lines = _read_stream_lines(shown_run, max_lines=80)
    with st.expander("Live CLI Stream (same run)", expanded=True):
        if stream_lines:
            st.code("\n".join(stream_lines), language="text")
        else:
            st.caption("No live stream lines yet for this run.")
    run_analysis = analyze_run(trace, feedback_text="", previous_runs=[])
    summary = run_analysis.get("summary", {}) if isinstance(run_analysis, dict) else {}
    tool_table = summary.get("tool_table", []) if isinstance(summary, dict) else []

    st.markdown("### Tool Scores and Recommendations")
    if tool_table:
        st.dataframe(tool_table, use_container_width=True, hide_index=True)
    else:
        st.info("No tool analysis available yet for this run.")

    with st.expander("Logs (optional)", expanded=False):
        for tool in blocks:
            rows = [e for e in events if e.get("tool") == tool]
            tool_steps = [s for s in steps if s.get("tool_name") == tool]
            with st.container(border=True):
                st.markdown(f"**`{tool}` logs ({len(rows)} events / {len(tool_steps)} steps)**")
                for e in rows[-40:]:
                    st.caption(
                        f"[{e.get('time')}] {e.get('event')} | {e.get('message','')} | {e.get('duration_ms',0)} ms"
                    )
                if tool_steps:
                    compact = []
                    for s in tool_steps[-8:]:
                        compact.append(
                            {
                                "step_id": s.get("step_id"),
                                "status": s.get("status"),
                                "latency_ms": round(float(s.get("latency_ms", 0.0)), 2),
                                "dependencies": s.get("dependencies", []),
                                "input_preview": _short_preview(s.get("input", "")),
                                "output_preview": _short_preview(s.get("output", "")),
                            }
                        )
                    st.code(json.dumps(compact, ensure_ascii=True, indent=2), language="json")


def render_dev_view() -> None:
    st.markdown(
        """
        <style>
          div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,0.3);
            border-radius: 12px;
            padding: 8px 10px;
            background: rgba(15,23,42,0.2);
          }
          div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 12px;
          }
          div[data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 12px;
            overflow: hidden;
          }
          div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
            margin-top: 0.3rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    rid = st.query_params.get("rid", "")
    st.title("Developer Observability")
    _render_dev_live_fragment(rid)
    st.markdown("---")
    st.subheader("Run History Insights")
    run_ids = _list_recent_run_ids(limit=30)
    if not run_ids:
        st.caption("No historical runs found yet.")
        return

    st.markdown("### Accumulated Tool Scores (Recent Runs)")
    accum = _accumulated_tool_scores(run_ids)
    if accum:
        st.dataframe(accum, use_container_width=True, hide_index=True)
    else:
        st.caption("No accumulated scores available yet.")

    selected_run = st.selectbox(
        "Inspect a past run (insights do not disappear)",
        options=run_ids,
        index=0,
        key="dev_history_run_selector",
    )
    hist_trace = _load_run_trace(selected_run)
    hist_analysis = analyze_run(hist_trace, feedback_text="", previous_runs=[])
    hist_summary = hist_analysis.get("summary", {}) if isinstance(hist_analysis, dict) else {}
    st.caption(f"Selected run: {selected_run}")

    pipeline = hist_summary.get("pipeline", []) if isinstance(hist_summary, dict) else []
    if pipeline:
        st.caption("Pipeline: " + " -> ".join(pipeline))
    st.caption(f"Primary issue: {hist_summary.get('primary_issue', 'unknown')}")

    hist_table = hist_summary.get("tool_table", []) if isinstance(hist_summary, dict) else []
    if hist_table:
        st.dataframe(hist_table, use_container_width=True, hide_index=True)
    else:
        st.caption("No tool table available for selected run.")

    with st.expander("Selected Run Logs (optional)", expanded=False):
        st.code(json.dumps(hist_trace.get("steps", []), ensure_ascii=True, indent=2, default=str), language="json")


def _open_dev_window_once(force: bool = False) -> None:
    if st.session_state.get("dev_window_opened") and not force:
        return
    if not force:
        st.session_state["dev_window_opened"] = True
    rid = st.session_state["run_id"]
    script = f"""
    <script>
    try {{
      const hostWindow = window.top || window.parent || window;
      const base = new URL(hostWindow.location.href);
      base.searchParams.set("view", "dev");
      base.searchParams.set("rid", "{rid}");
      hostWindow.open(base.toString(), "_blank", "noopener,noreferrer,width=1280,height=900");
    }} catch (e) {{
      console.error(e);
    }}
    </script>
    """
    components.html(script, height=0)


def save_main_file(uploaded_file) -> str:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    for f in MAIN_DIR.iterdir():
        if f.is_file():
            f.unlink()
    path = MAIN_DIR / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


def save_supporting_files(files) -> list[str]:
    SUPPORTING_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for f in files:
        target = SUPPORTING_DIR / f.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            i = 1
            while target.exists():
                target = SUPPORTING_DIR / f"{stem}_{i}{suffix}"
                i += 1
        target.write_bytes(f.getbuffer())
        paths.append(str(target))
    return paths


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
    for k, v in structure.items():
        path = f"{parent} > {k}" if parent else k
        out.append(path)
        if isinstance(v, dict) and v:
            out.extend(flatten_structure_keys(v, path))
    return out


def structure_report_to_markdown(structure: dict, report: dict[str, str], parent: str = "", level: int = 1) -> str:
    parts: list[str] = []
    for section, children in structure.items():
        path = f"{parent} > {section}" if parent else section
        heading = "#" * max(1, min(level, 6))
        parts.append(f"{heading} {section}")
        parts.append((report.get(path, "") or "").strip())
        if isinstance(children, dict) and children:
            sub = structure_report_to_markdown(children, report, path, level + 1)
            if sub.strip():
                parts.append(sub)
    return "\n\n".join(p for p in parts if p is not None).strip()


def markdown_to_structure_report(markdown_text: str) -> tuple[dict, dict[str, str]]:
    lines = markdown_text.splitlines()
    root: dict = {}
    report: dict[str, str] = {}
    stack: list[tuple[int, str, dict]] = []
    current_path: str | None = None
    acc: list[str] = []

    def flush() -> None:
        nonlocal acc
        if current_path is not None:
            report[current_path] = "\n".join(acc).strip()
        acc = []

    for raw in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", raw.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                root.setdefault(title, {})
                node = root[title]
                path = title
            else:
                _, ppath, pnode = stack[-1]
                pnode.setdefault(title, {})
                node = pnode[title]
                path = f"{ppath} > {title}"
            stack.append((level, path, node))
            current_path = path
        else:
            acc.append(raw)
    flush()
    return root, report


def _extract_dict_from_text(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
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


def _normalize_structure(candidate) -> dict:
    if not isinstance(candidate, dict):
        return {}
    out: dict = {}
    for k, v in candidate.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = _normalize_structure(v) if isinstance(v, dict) else {}
    return out


def _heuristic_structure_from_chunks(chunks: list[dict]) -> dict:
    text = " ".join((c.get("text", "") or "") for c in chunks).lower()
    rules = [
        ("Concepts and Definitions", ["define", "definition", "concept", "framework", "principle"]),
        ("Methods and Approaches", ["method", "approach", "process", "analysis", "model"]),
        ("Key Findings", ["result", "finding", "impact", "evidence", "outcome"]),
        ("Challenges and Limitations", ["challenge", "barrier", "risk", "limitation", "constraint"]),
        ("Recommendations", ["recommend", "strategy", "policy", "action", "improve"]),
    ]
    picked = [name for name, kws in rules if sum(text.count(k) for k in kws) >= 2]
    if not picked:
        picked = ["Concepts and Definitions", "Methods and Approaches", "Key Findings"]
    s: dict = {"Introduction": {}}
    for name in picked[:4]:
        s[name] = {}
    s["Conclusion"] = {}
    return s


def _structure_relevance_score(
    structure: dict, chunks: list[dict], embeddings, retrieve_top_k_scored
) -> float:
    sections = flatten_structure_keys(structure)
    if not sections or not chunks:
        return 0.0

    # Relevance = fraction of supporting-doc chunks that can be organized
    # under at least one structure section with a meaningful match score.
    covered_chunk_keys: set[str] = set()
    per_section_k = max(4, min(20, len(chunks) // max(1, len(sections)) + 2))
    min_match_score = 0.18

    for sec in sections:
        scored = retrieve_top_k_scored(sec, chunks, embeddings, top_k=per_section_k)
        for ch, score in scored:
            if score < min_match_score:
                continue
            key = f"{ch.get('source','')}|{ch.get('page','')}|{(ch.get('text','') or '')[:120]}"
            covered_chunk_keys.add(key)

    return len(covered_chunk_keys) / max(1, len(chunks))


def _propose_structure_from_docs(chunks: list[dict], original_structure: dict) -> dict:
    sample = "\n\n".join(
        f"[{c.get('source','')} | p{c.get('page','')}]\n{(c.get('text','') or '')[:350]}"
        for c in chunks[:10]
    )
    prompt = f"""
Build a report structure from supporting-document evidence.
Current structure:
{json.dumps(original_structure, indent=2)}

Evidence:
{sample}

Return STRICT JSON dict only.
Use format: {{"Section": {{}}, "Parent": {{"Child": {{}}}}}}
"""
    resp = STRUCTURE_LLM.invoke(prompt)
    txt = resp.content if isinstance(resp.content, str) else str(resp.content)
    parsed = _normalize_structure(_extract_dict_from_text(txt))
    if parsed and len(flatten_structure_keys(parsed)) >= 4:
        return parsed
    return _heuristic_structure_from_chunks(chunks)


def generate_report(structure: dict, supporting_paths: list[str]) -> dict[str, str]:
    from main import (
        filter_chunks,
        generate_section,
        get_or_build_chunk_index,
        retrieve_top_k,
        retrieve_top_k_scored,
    )

    (read_output, s_read_supporting) = _run_tool(
        "read_supporting",
        get_or_build_chunk_index,
        dependencies=[],
        input_data={"supporting_paths": supporting_paths},
        document_paths=supporting_paths,
    )
    chunks, embeddings, _ = read_output
    (relevance_score, s_check_structure) = _run_tool(
        "check_structure",
        _structure_relevance_score,
        dependencies=[s_read_supporting],
        input_data={"sections": len(flatten_structure_keys(structure)), "chunks_count": len(chunks)},
        structure=structure,
        chunks=chunks,
        embeddings=embeddings,
        retrieve_top_k_scored=retrieve_top_k_scored,
    )
    _append_event(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "run_id": st.session_state["run_id"],
            "tool": "check_structure",
            "event": "info",
            "message": f"structure_coverage_ratio={relevance_score:.3f} (threshold=0.800)",
            "duration_ms": 0,
        }
    )
    final_structure = structure
    # If relevance is low, current structure is not reliable for RAG retrieval.
    if relevance_score < 0.8:
        (final_structure, s_propose) = _run_tool(
            "propose_structure",
            _propose_structure_from_docs,
            dependencies=[s_check_structure, s_read_supporting],
            input_data={"chunks_count": len(chunks)},
            chunks=chunks,
            original_structure=structure,
        )
        st.session_state["structure"] = final_structure
        st.session_state["structure_origin"] = "proposed"
        _append_event(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "run_id": st.session_state["run_id"],
                "tool": "propose_structure",
                "event": "info",
                "message": "Using proposed structure for report generation",
                "duration_ms": 0,
            }
        )
    else:
        st.session_state["structure_origin"] = "original"
        _append_event(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "run_id": st.session_state["run_id"],
                "tool": "propose_structure",
                "event": "skipped",
                "message": f"Skipped: structure coverage is high ({relevance_score:.3f})",
                "duration_ms": 0,
            }
        )

    sections = flatten_structure_keys(final_structure)
    report: dict[str, str] = {}
    dep_structure = s_propose if relevance_score < 0.8 else s_check_structure
    for section in sections:
        retrieved_scored, s_retrieve_scored = _run_tool(
            "retrieve",
            retrieve_top_k_scored,
            dependencies=[dep_structure, s_read_supporting],
            input_data={"section": section, "top_k": 3, "scored": True},
            section_key=section,
            chunks=chunks,
            chunk_embeddings=embeddings,
            top_k=3,
        )
        strong = [ch for ch, score in retrieved_scored if score >= 0.20]
        retrieved = strong if strong else [ch for ch, _ in retrieved_scored]
        dep_retrieve = s_retrieve_scored
        if not retrieved:
            retrieved, s_retrieve = _run_tool(
                "retrieve",
                retrieve_top_k,
                dependencies=[s_retrieve_scored],
                input_data={"section": section, "top_k": 3, "fallback": True},
                section_key=section,
                chunks=chunks,
                chunk_embeddings=embeddings,
                top_k=3,
            )
            dep_retrieve = s_retrieve
        selected = filter_chunks(retrieved, top_k=3)
        existing = "\n\n".join(v for v in report.values() if v)
        generated, _ = _run_tool(
            "generate",
            generate_section,
            dependencies=[dep_retrieve],
            input_data={"section": section, "selected_count": len(selected)},
            section_key=section,
            selected_chunks=selected,
            existing_report_text=existing,
        )
        report[section] = generated
    return report


def _find_best_section(query: str, sections: list[str]) -> str | None:
    if not sections:
        return None
    low = (query or "").lower().strip()
    for s in sections:
        sl = s.lower()
        if sl in low or low in sl:
            return s
    scored = [(SequenceMatcher(None, low, s.lower()).ratio(), s) for s in sections]
    score, best = max(scored, key=lambda x: x[0])
    return best if score >= 0.25 else sections[0]


def _is_edit_request(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(
        k in low
        for k in [
            "improve",
            "edit",
            "update",
            "rewrite",
            "expand",
            "add more",
            "add section",
            "fix",
            "change",
            "revise",
            "make it better",
            "elaborate",
        ]
    )


def _route_chat_prompt(prompt: str) -> str:
    """
    Deterministic lightweight router for non-edit chat prompts.
    Returns one of:
      - chat_read
      - chat_retrieve
      - chat_read_supporting_docs
      - chat_rag_pipeline
      - chat_generate (default)
    """
    low = (prompt or "").lower().strip()
    if any(k in low for k in ["supporting document", "source", "citation", "evidence", "pdf"]):
        return "chat_read_supporting_docs"
    if any(k in low for k in ["rag", "regenerate", "rebuild section", "new section from docs"]):
        return "chat_rag_pipeline"
    if any(k in low for k in ["retrieve", "relevant snippet", "which part", "find where"]):
        return "chat_retrieve"
    if any(k in low for k in ["show report", "read section", "what is in section", "display section"]):
        return "chat_read"
    return "chat_generate"


def _extract_new_section_name(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"', prompt) + re.findall(r"'([^']+)'", prompt)
    if quoted:
        return quoted[0].strip()
    m = re.search(r"add section\s*[:\-]?\s*(.+)$", prompt, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().title()
    return "New Section"


def _apply_chat_edit(prompt: str) -> str:
    report = st.session_state.get("report", {})
    if not report:
        return "No report available to edit yet."

    low = prompt.lower()
    if "add section" in low:
        name = _extract_new_section_name(prompt)
        ctx = "\n\n".join(f"{k}\n{v}" for k, v in report.items())[:4200]
        gen_prompt = f"Add a new report section named '{name}'. Use context below and keep one clear paragraph.\n\n{ctx}"
        resp, _ = _run_tool(
            "chat_edit",
            CHAT_LLM.invoke,
            dependencies=[],
            input_data={"request": prompt, "mode": "add_section"},
            input=gen_prompt,
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        report[name] = text.strip()
        st.session_state["report"] = report
        return f"Added section `{name}`."

    section = _find_best_section(prompt, list(report.keys()))
    current = report.get(section or "", "")
    rewrite_prompt = f"""
You are editing a report section.
Section: {section}
User request: {prompt}
Current text: {current}
Return only the revised section paragraph.
"""
    resp, _ = _run_tool(
        "chat_edit",
        CHAT_LLM.invoke,
        dependencies=[],
        input_data={"request": prompt, "target_section": section},
        input=rewrite_prompt,
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    if section:
        report[section] = text.strip()
        st.session_state["report"] = report
        return f"Updated `{section}`."
    return "I could not find a section to update."


def _get_chat_agent():
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    def tool_read(query: str) -> str:
        report = st.session_state.get("report", {})
        if not report:
            return "No report generated yet."
        if not query or query.strip().lower() in {"all", "overview", "report"}:
            return "\n\n".join(f"{k}\n{v}" for k, v in report.items())
        section = _find_best_section(query, list(report.keys()))
        return f"{section}\n\n{report.get(section, '')}" if section else "Section not found."

    def tool_retrieve(query: str) -> str:
        report = st.session_state.get("report", {})
        section = _find_best_section(query, list(report.keys()))
        if not section:
            return "No related section found."
        text = report.get(section, "")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return f"{section}\n" + " ".join(sents[:3])

    def tool_generate(query: str) -> str:
        report = st.session_state.get("report", {})
        context = "\n\n".join(f"{k}\n{v}" for k, v in report.items())[:4800]
        prompt = f"""
You are a helpful report assistant.
Use only report context.
Answer naturally and clearly.

Context:
{context}

User:
{query}
"""
        resp = CHAT_LLM.invoke(prompt)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    def tool_read_supporting(query: str) -> str:
        from main import get_or_build_chunk_index, retrieve_top_k_scored

        supporting_paths = st.session_state.get("supporting_paths", [])
        if not supporting_paths:
            return "No supporting documents are available."
        (read_out, s_read) = _run_tool(
            "read_supporting",
            get_or_build_chunk_index,
            dependencies=[],
            input_data={"supporting_paths": supporting_paths, "mode": "chat_read_supporting_docs"},
            document_paths=supporting_paths,
        )
        chunks, embeddings, _ = read_out
        if not chunks:
            return "No readable chunks were extracted from supporting documents."
        scored, _ = _run_tool(
            "retrieve",
            retrieve_top_k_scored,
            dependencies=[s_read],
            input_data={"query": (query or "overview").strip(), "top_k": 4, "mode": "chat_read_supporting_docs"},
            section_key=(query or "overview").strip(),
            chunks=chunks,
            chunk_embeddings=embeddings,
            top_k=4,
        )
        if not scored:
            return "No relevant supporting-document chunks were found."
        lines: list[str] = []
        for ch, score in scored[:4]:
            snippet = re.sub(r"\s+", " ", ch.get("text", "")).strip()
            if len(snippet) > 240:
                snippet = snippet[:240].rsplit(" ", 1)[0] + "..."
            lines.append(
                f"- [{ch.get('source','?')} | p{ch.get('page','?')} | score={score:.2f}] {snippet}"
            )
        return "\n".join(lines)

    def tool_rag_pipeline(query: str) -> str:
        from main import filter_chunks, generate_section, get_or_build_chunk_index, retrieve_top_k_scored

        supporting_paths = st.session_state.get("supporting_paths", [])
        if not supporting_paths:
            return "No supporting documents are available for RAG."
        (read_out, s_read) = _run_tool(
            "read_supporting",
            get_or_build_chunk_index,
            dependencies=[],
            input_data={"supporting_paths": supporting_paths, "mode": "chat_rag_pipeline"},
            document_paths=supporting_paths,
        )
        chunks, embeddings, _ = read_out
        if not chunks:
            return "No supporting chunks available for RAG."
        scored, s_retrieve = _run_tool(
            "retrieve",
            retrieve_top_k_scored,
            dependencies=[s_read],
            input_data={"query": (query or "summary").strip(), "top_k": 4, "mode": "chat_rag_pipeline"},
            section_key=(query or "summary").strip(),
            chunks=chunks,
            chunk_embeddings=embeddings,
            top_k=4,
        )
        retrieved = [ch for ch, _ in scored]
        selected = filter_chunks(retrieved, top_k=3)
        existing = "\n\n".join(v for v in st.session_state.get("report", {}).values() if v)
        generated, _ = _run_tool(
            "generate",
            generate_section,
            dependencies=[s_retrieve],
            input_data={"section": (query or "General Update").strip(), "selected_count": len(selected), "mode": "chat_rag_pipeline"},
            section_key=(query or "General Update").strip(),
            selected_chunks=selected,
            existing_report_text=existing,
        )
        return generated

    tools = [
        Tool(
            name="read",
            func=lambda x: _run_tool(
                "chat_read", tool_read, dependencies=[], input_data={"query": x}, query=x
            )[0],
            description="Read report content.",
        ),
        Tool(
            name="retrieve",
            func=lambda x: _run_tool(
                "chat_retrieve", tool_retrieve, dependencies=[], input_data={"query": x}, query=x
            )[0],
            description="Retrieve relevant snippets.",
        ),
        Tool(
            name="generate",
            func=lambda x: _run_tool(
                "chat_generate", tool_generate, dependencies=[], input_data={"query": x}, query=x
            )[0],
            description="Generate final response from report context when user asks a question.",
        ),
        Tool(
            name="read_supporting_docs",
            func=lambda x: _run_tool(
                "chat_read_supporting_docs", tool_read_supporting, dependencies=[], input_data={"query": x}, query=x
            )[0],
            description="Read supporting-document evidence. Use when user asks for source-backed document snippets.",
        ),
        Tool(
            name="rag_pipeline",
            func=lambda x: _run_tool(
                "chat_rag_pipeline", tool_rag_pipeline, dependencies=[], input_data={"query": x}, query=x
            )[0],
            description="Run retrieval + filtering + section generation on supporting docs for new RAG-backed content.",
        ),
    ]
    _AGENT = initialize_agent(
        tools=tools,
        llm=CHAT_LLM,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    return _AGENT


def render_upload_page() -> None:
    st.title("Upload Documents")
    main_file = st.file_uploader("Upload structure file", type=ALLOWED_TYPES, key="main")
    supporting_files = st.file_uploader(
        "Upload supporting documents",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="supporting",
    )

    if st.button("Generate Report"):
        if not main_file:
            st.error("Please upload one structure file.")
            return
        if not supporting_files:
            st.error("Please upload at least one supporting document.")
            return

        _reset_run_trace(uuid.uuid4().hex[:12])
        st.session_state["dev_window_opened"] = False
        _open_dev_window_once()

        main_path = save_main_file(main_file)
        supporting_paths = save_supporting_files(supporting_files)
        structure_text, _ = _run_tool(
            "read_structure",
            read_file_as_text,
            dependencies=[],
            input_data={"structure_path": main_path},
            file_path=main_path,
        )
        structure = parse_inline_structure(structure_text)

        with st.spinner("Processing supporting documents and generating report..."):
            report = generate_report(structure, supporting_paths)

        # structure may have been auto-updated by generate_report
        st.session_state["structure"] = st.session_state.get("structure") or structure
        if not st.session_state["structure"]:
            st.session_state["structure"] = structure
        st.session_state["report"] = report
        st.session_state["messages"] = []
        st.session_state["supporting_paths"] = supporting_paths
        st.session_state["page"] = "workspace"
        st.rerun()


def render_workspace_page() -> None:
    report = st.session_state.get("report", {})
    if not report:
        st.warning("No report yet. Upload documents first.")
        return

    st.title("Report Workspace")
    if st.session_state.get("structure_origin") == "proposed":
        st.info("Structure source: proposed from supporting documents (used for this report).")
    else:
        st.caption("Structure source: original uploaded structure.")
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Editable Report (single markdown textbox)")
        structure = st.session_state.get("structure", {})
        current_md = structure_report_to_markdown(structure, report)
        if "report_markdown_editor" not in st.session_state:
            st.session_state["report_markdown_editor"] = current_md
            st.session_state["_report_md_sig"] = hash(current_md)
        elif st.session_state.get("_report_md_sig") != hash(current_md):
            st.session_state["report_markdown_editor"] = current_md
            st.session_state["_report_md_sig"] = hash(current_md)
        if st.session_state.get("chat_locked") and st.session_state.get("_report_md_sig") == hash(current_md):
            st.session_state["chat_locked"] = False

        st.text_area(
            "Full report editor (use markdown headings: # / ##)",
            key="report_markdown_editor",
            height=560,
        )
        if st.button("Apply Report Edits"):
            new_structure, parsed_report = markdown_to_structure_report(
                st.session_state["report_markdown_editor"]
            )
            if not new_structure:
                st.error("No headings found. Add headings like # Section.")
            else:
                ordered: dict[str, str] = {}
                for key in flatten_structure_keys(new_structure):
                    ordered[key] = parsed_report.get(key, "")
                st.session_state["structure"] = new_structure
                st.session_state["report"] = ordered
                st.session_state["_report_md_sig"] = hash(
                    structure_report_to_markdown(new_structure, ordered)
                )
                st.success("Report and structure updated.")
                st.rerun()

    with right:
        st.subheader("Chat")
        if st.button("Open Developer View", use_container_width=True):
            _open_dev_window_once(force=True)
        messages = st.session_state.setdefault("messages", [])
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        if st.session_state.get("chat_locked"):
            st.caption("Applying previous change to report editor. Please wait...")
        prompt = st.chat_input(
            "Ask questions or request edits...",
            disabled=bool(st.session_state.get("chat_locked")),
        )
        if prompt:
            st.session_state["chat_locked"] = True
            messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                if _is_edit_request(prompt):
                    _append_event(
                        {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "run_id": st.session_state["run_id"],
                            "tool": "chat",
                            "event": "skipped",
                            "message": "Skipped: edit request routed to chat_edit",
                            "duration_ms": 0,
                        }
                    )
                    response = _apply_chat_edit(prompt)
                else:
                    _append_event(
                        {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "run_id": st.session_state["run_id"],
                            "tool": "chat_edit",
                            "event": "skipped",
                            "message": "Skipped: non-edit request routed to chat",
                            "duration_ms": 0,
                        }
                    )
                    agent = _get_chat_agent()
                    route_tool_name = _route_chat_prompt(prompt)
                    tool_by_name = {t.name: t for t in agent.tools}
                    try:
                        # Prefer deterministic route for separable tool execution/logging.
                        if route_tool_name == "chat_read" and "read" in tool_by_name:
                            response = tool_by_name["read"].func(prompt)
                        elif route_tool_name == "chat_retrieve" and "retrieve" in tool_by_name:
                            response = tool_by_name["retrieve"].func(prompt)
                        elif route_tool_name == "chat_read_supporting_docs" and "read_supporting_docs" in tool_by_name:
                            response = tool_by_name["read_supporting_docs"].func(prompt)
                        elif route_tool_name == "chat_rag_pipeline" and "rag_pipeline" in tool_by_name:
                            response = tool_by_name["rag_pipeline"].func(prompt)
                        elif route_tool_name == "chat_generate" and "generate" in tool_by_name:
                            response = tool_by_name["generate"].func(prompt)
                        else:
                            response, _ = _run_tool(
                                "chat",
                                lambda prompt_text: agent.run(prompt_text),
                                dependencies=[],
                                input_data={"prompt": prompt},
                                prompt_text=prompt,
                            )
                    except Exception:
                        response, _ = _run_tool(
                            "chat",
                            lambda prompt_text: agent.tools[2].func(prompt_text),
                            dependencies=[],
                            input_data={"prompt": prompt, "fallback": True},
                            prompt_text=prompt,
                        )
            messages.append({"role": "assistant", "content": response})
            st.session_state["messages"] = messages
            st.rerun()


_init_state()
view = st.query_params.get("view", "main")
if view == "dev":
    render_dev_view()
else:
    if st.session_state.get("page") == "upload":
        render_upload_page()
    else:
        render_workspace_page()
