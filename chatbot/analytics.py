"""
analytics.py
============

Log and analyse usage of the VI Copilot chatbot.

Every question gets one JSON line in output/analytics.jsonl:
    {"ts": "2026-09-01T10:23:45", "question": "...", "engine": "rule-based",
     "model": "rule-based (no LLM)", "ms": 42, "plans": ["26106058"]}

The file is git-ignored (runtime data, machine-local). On Streamlit Cloud it
resets when the instance restarts, but persists across sessions for the same
running instance -- good enough for a demo or a short monitoring window.

All analysis functions read from the JSONL file and return plain Python
structures, so the analytics page (pages/2_Analytics.py) is a thin display
layer with no logic of its own.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "output" / "analytics.jsonl"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def log_event(question: str, engine: str, model: str = "",
              ms: int = 0, answer: str = "") -> None:
    """
    Append one usage event. Never raises -- a logging failure must never crash
    the answer path. The file and its parent are created if they don't exist.
    """
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        plans = re.findall(r"\b\d{8}\b", answer)
        event = {
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "question": question.strip(),
            "engine": engine,
            "model": model or engine,
            "ms": ms,
            "plans": list(dict.fromkeys(plans)),  # deduplicated, order-preserved
        }
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Reading events
# --------------------------------------------------------------------------

def _iter_events() -> Iterator[dict]:
    if not LOG_FILE.exists():
        return
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_events() -> list[dict]:
    return list(_iter_events())


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def summary(events: list[dict]) -> dict:
    """
    Headline numbers for the top of the analytics page.
    """
    if not events:
        return {"total": 0, "today": 0, "rule_pct": 0, "gemini_pct": 0,
                "avg_ms": 0, "unique_questions": 0}

    today_str = date.today().isoformat()
    today_count = sum(1 for e in events if e.get("ts", "").startswith(today_str))
    rule_count = sum(1 for e in events if "rule" in e.get("engine", "").lower())
    gemini_count = len(events) - rule_count
    total = len(events)
    times = [e["ms"] for e in events if e.get("ms")]
    return {
        "total": total,
        "today": today_count,
        "rule_pct": round(rule_count * 100 / total),
        "gemini_pct": round(gemini_count * 100 / total),
        "avg_ms": round(sum(times) / len(times)) if times else 0,
        "unique_questions": len({e["question"].lower() for e in events}),
    }


def top_questions(events: list[dict], n: int = 10) -> list[dict]:
    """
    Most-asked questions, case-insensitive, with their count.
    Returns [{"question": str, "count": int, "engine": str (most common engine)}]
    """
    from collections import defaultdict
    buckets: dict[str, list[str]] = defaultdict(list)
    for e in events:
        key = e["question"].strip().lower()
        buckets[key].append(e.get("engine", ""))

    rows = []
    for q_lower, engines in sorted(buckets.items(), key=lambda x: -len(x[1])):
        most_common_engine = Counter(engines).most_common(1)[0][0] if engines else ""
        # Recover original casing from the first occurrence.
        original = next(
            (e["question"] for e in events if e["question"].lower() == q_lower), q_lower
        )
        rows.append({"question": original, "count": len(engines),
                     "engine": most_common_engine})

    return rows[:n]


def top_plans(events: list[dict], n: int = 10) -> list[dict]:
    """
    Most-cited plan codes across all answers.
    Returns [{"code": str, "count": int}]
    """
    counter: Counter = Counter()
    for e in events:
        for code in e.get("plans", []):
            counter[code] += 1
    return [{"code": code, "count": cnt} for code, cnt in counter.most_common(n)]


def engine_breakdown(events: list[dict]) -> dict[str, int]:
    """
    Rule-based vs Gemini-powered question counts.
    """
    breakdown: dict[str, int] = {}
    for e in events:
        key = e.get("engine", "unknown")
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


def daily_activity(events: list[dict]) -> list[dict]:
    """
    Questions per calendar day, sorted oldest first.
    Returns [{"date": "YYYY-MM-DD", "count": int}]
    """
    by_day: dict[str, int] = {}
    for e in events:
        day = e.get("ts", "")[:10]
        if day:
            by_day[day] = by_day.get(day, 0) + 1
    return [{"date": d, "count": c} for d, c in sorted(by_day.items())]


def clear_log() -> int:
    """Delete all log entries. Returns the number of events that were cleared."""
    if not LOG_FILE.exists():
        return 0
    events = load_events()
    count = len(events)
    LOG_FILE.unlink()
    return count
