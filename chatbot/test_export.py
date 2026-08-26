"""
test_export.py
==============

Offline tests for the conversation / comparison exporters (export.py).

The export is what a rep actually keeps after a call, so the things worth
pinning down are: nothing gets silently dropped, the provenance header is
present, and an empty conversation degrades gracefully instead of producing a
misleading blank document.

Run:  python -m chatbot.test_export
"""

from __future__ import annotations

import sys
from datetime import datetime

from .export import (comparison_markdown, conversation_markdown,
                     suggested_filename)

# Fixed timestamp so the expected strings never depend on the clock.
WHEN = datetime(2026, 8, 26, 14, 30, 5)

MESSAGES = [
    {"role": "user", "content": "What is the price of plan 26106058?"},
    {"role": "assistant", "content": "**Vi Hero Unlimited 449** — price: ₹449"},
    {"role": "user", "content": "compare it with 299"},
    {"role": "assistant", "content": "| Feature | A | B |\n| --- | --- | --- |"},
]


def build_cases() -> list[tuple[str, object]]:
    """(label, check) where check() returns None on success or a reason."""
    full = conversation_markdown(MESSAGES, mode="Rule-based (no LLM)",
                                 model="rule-based", plan_count=62, when=WHEN)
    empty = conversation_markdown([], mode="Gemini-powered", when=WHEN)
    comparison = comparison_markdown("| Feature | A |\n| --- | --- |",
                                     ["Vi Hero 449", "Vi Hero 299"], when=WHEN)

    def has(text, needle, what):
        return None if needle in text else f"{what}: missing {needle!r}"

    return [
        ("title present", lambda: has(full, "# VI Sales Copilot", "conversation")),
        ("export timestamp formatted",
         lambda: has(full, "26 Aug 2026, 14:30", "conversation")),
        ("engine recorded", lambda: has(full, "Rule-based (no LLM)", "conversation")),
        ("model recorded", lambda: has(full, "`rule-based`", "conversation")),
        ("catalogue size recorded", lambda: has(full, "62 plans", "conversation")),

        ("roles renamed for humans",
         lambda: has(full, "### Rep", "conversation") or has(full, "### Copilot",
                                                             "conversation")),
        ("raw role names not leaked",
         lambda: "### user" not in full and "### assistant" not in full
         or "raw role name leaked into transcript"),

        # The whole point of an export is that nothing is lost.
        ("every message body survives",
         lambda: next((f"dropped: {m['content'][:30]!r}" for m in MESSAGES
                       if m["content"].strip() not in full), None)),
        ("markdown tables survive intact",
         lambda: has(full, "| Feature | A | B |", "conversation")),
        ("rupee symbol survives", lambda: has(full, "₹449", "conversation")),

        ("empty conversation says so, not blank",
         lambda: has(empty, "No messages in this conversation yet", "empty export")),
        ("empty conversation still carries the header",
         lambda: has(empty, "Gemini-powered", "empty export")),

        ("comparison doc has its own title",
         lambda: has(comparison, "# VI plan comparison", "comparison")),
        ("comparison lists the plans compared",
         lambda: has(comparison, "Vi Hero 449, Vi Hero 299", "comparison")),
        ("comparison keeps the table",
         lambda: has(comparison, "| Feature | A |", "comparison")),

        ("provenance footer on both docs",
         lambda: has(full, "loaded VI catalogue", "conversation")
         or has(comparison, "loaded VI catalogue", "comparison")),

        ("filename is timestamped",
         lambda: has(suggested_filename(when=WHEN), "20260826-143005", "filename")),
        ("filename honours the prefix and extension",
         lambda: None if suggested_filename(prefix="vi-comparison", when=WHEN)
         == "vi-comparison-20260826-143005.md"
         else f"got {suggested_filename(prefix='vi-comparison', when=WHEN)}"),
        ("repeated exports never collide",
         lambda: None if suggested_filename(when=WHEN)
         != suggested_filename(when=datetime(2026, 8, 26, 14, 30, 6))
         else "same filename for different seconds"),
    ]


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = build_cases()
    passed = 0
    print(f"Running {len(cases)} export tests...\n")
    for label, check in cases:
        try:
            reason = check()
            # A check may return True (bare boolean style) -- treat as pass.
            reason = None if reason is True else reason
        except Exception as exc:
            reason = f"raised {type(exc).__name__}: {exc}"
        passed += reason is None
        print(f"  [{'PASS' if reason is None else 'FAIL'}] {label}")
        if reason:
            print(f"         {reason}")

    print(f"\nRESULT: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(run())
