"""
test_analytics.py
=================

Offline tests for the analytics engine (analytics.py). No Streamlit, no network.
All file I/O is redirected to a temp directory so the live log is never touched.

Run:  python -m chatbot.test_analytics
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import analytics as al


def _sandbox(fn):
    """Run fn with LOG_FILE pointing at a temp file; restore afterwards."""
    saved = al.LOG_FILE
    with tempfile.TemporaryDirectory() as tmp:
        al.LOG_FILE = Path(tmp) / "analytics.jsonl"
        try:
            return fn()
        finally:
            al.LOG_FILE = saved


class Check:
    def __init__(self, label, fn):
        self.label, self.fn = label, fn


def _log_events_checks() -> list[Check]:
    def log_creates_file():
        def body():
            al.log_event("price of 449?", "Rule-based (no LLM)", ms=10, answer="₹449")
            return al.LOG_FILE.exists()
        return _sandbox(body)

    def log_multiple_events():
        def body():
            al.log_event("q1", "Rule-based (no LLM)", ms=5, answer="a")
            al.log_event("q2", "Gemini-powered", ms=300, answer="b")
            return len(al.load_events()) == 2
        return _sandbox(body)

    def log_captures_plan_codes():
        def body():
            al.log_event("plan?", "Rule-based (no LLM)", ms=10,
                         answer="Plan 26106058 costs ₹449. See also 26104439.")
            e = al.load_events()[0]
            return set(e["plans"]) == {"26106058", "26104439"}
        return _sandbox(body)

    def log_never_raises_on_bad_path():
        saved = al.LOG_FILE
        al.LOG_FILE = Path("/nonexistent_dir/sub/analytics.jsonl")
        try:
            al.log_event("q", "engine")  # must not raise
            return True
        except Exception:
            return False
        finally:
            al.LOG_FILE = saved

    def load_skips_corrupt_lines():
        def body():
            al.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            al.LOG_FILE.write_text('{"ts":"2026-09-01T10:00:00","question":"ok","engine":"rule-based","model":"rule-based","ms":1,"plans":[]}\nNOT JSON\n', encoding="utf-8")
            return len(al.load_events()) == 1
        return _sandbox(body)

    return [
        Check("log_event creates the file", log_creates_file),
        Check("multiple events stored in order", log_multiple_events),
        Check("8-digit plan codes extracted from answer", log_captures_plan_codes),
        Check("log_event never raises even on bad path", log_never_raises_on_bad_path),
        Check("load_events skips corrupt lines", load_skips_corrupt_lines),
    ]


def _analysis_checks() -> list[Check]:
    EVENTS = [
        {"ts": "2026-09-01T09:00:00", "question": "price of 449?",
         "engine": "Rule-based (no LLM)", "model": "rule-based", "ms": 10, "plans": ["26106058"]},
        {"ts": "2026-09-01T09:05:00", "question": "price of 449?",
         "engine": "Gemini-powered", "model": "gemini-3.6-flash", "ms": 800, "plans": ["26106058"]},
        {"ts": "2026-09-01T09:10:00", "question": "5G plans under 400",
         "engine": "Rule-based (no LLM)", "model": "rule-based", "ms": 15, "plans": []},
        {"ts": "2026-09-02T10:00:00", "question": "compare 299 and 449",
         "engine": "Gemini-powered", "model": "gemini-3.6-flash", "ms": 950, "plans": ["26104439", "26106058"]},
    ]

    def summary_totals():
        s = al.summary(EVENTS)
        return s["total"] == 4 and s["unique_questions"] == 3

    def summary_today_zero():
        s = al.summary(EVENTS)
        return s["today"] == 0  # all events are on 2026-09-01/02, not today

    def summary_engine_split():
        s = al.summary(EVENTS)
        return s["rule_pct"] == 50 and s["gemini_pct"] == 50

    def summary_avg_ms():
        s = al.summary(EVENTS)
        return s["avg_ms"] == round((10 + 800 + 15 + 950) / 4)

    def top_questions_ranking():
        top = al.top_questions(EVENTS, n=3)
        return top[0]["question"] == "price of 449?" and top[0]["count"] == 2

    def top_plans_ranking():
        plans = al.top_plans(EVENTS, n=5)
        codes = [p["code"] for p in plans]
        return codes[0] == "26106058" and plans[0]["count"] == 3

    def engine_breakdown_counts():
        bd = al.engine_breakdown(EVENTS)
        return bd.get("Rule-based (no LLM)") == 2 and bd.get("Gemini-powered") == 2

    def daily_activity_days():
        daily = al.daily_activity(EVENTS)
        dates = [d["date"] for d in daily]
        return dates == ["2026-09-01", "2026-09-02"] and daily[0]["count"] == 3

    def summary_empty():
        s = al.summary([])
        return s["total"] == 0 and s["avg_ms"] == 0

    def clear_log_removes_file():
        def body():
            al.log_event("q", "engine")
            assert al.LOG_FILE.exists()
            count = al.clear_log()
            return count == 1 and not al.LOG_FILE.exists()
        return _sandbox(body)

    return [
        Check("summary: total and unique counts", summary_totals),
        Check("summary: today count is 0 for old events", summary_today_zero),
        Check("summary: rule/gemini split", summary_engine_split),
        Check("summary: average response time", summary_avg_ms),
        Check("top_questions: most-asked ranked first", top_questions_ranking),
        Check("top_plans: most-mentioned code ranked first", top_plans_ranking),
        Check("engine_breakdown: per-engine counts", engine_breakdown_counts),
        Check("daily_activity: two days sorted oldest first", daily_activity_days),
        Check("summary: handles empty events list", summary_empty),
        Check("clear_log: removes file and returns count", clear_log_removes_file),
    ]


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    checks = _log_events_checks() + _analysis_checks()
    passed = 0
    print(f"Running {len(checks)} analytics tests...\n")
    for i, check in enumerate(checks, 1):
        try:
            ok = bool(check.fn())
            reason = ""
        except Exception as exc:
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        passed += ok
        print(f"[{i:>2}/{len(checks)}] {'PASS' if ok else 'FAIL'}  {check.label}")
        if not ok:
            print(f"        {reason}")

    print(f"\nRESULT: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(run())
