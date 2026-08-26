"""
run_tests.py
============

Run every offline test suite in one go and print a combined summary.

All four suites are offline and deterministic -- no API key, no network, no
quota -- so this is safe to run on every change:

    python run_tests.py

(chatbot/eval.py is deliberately NOT included: it calls the live Gemini API and
spends real free-tier quota, so it stays a separate, explicit command.)
"""

from __future__ import annotations

import sys

SUITES = [
    ("Rule engine + Plan Finder", "chatbot.test_rules"),
    ("Guardrails", "chatbot.test_guardrails"),
    ("Fallback chain", "chatbot.test_llm"),
    ("Export", "chatbot.test_export"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from importlib import import_module

    failed: list[str] = []
    for name, module_path in SUITES:
        print("=" * 62)
        print(f"  {name}   ({module_path})")
        print("=" * 62)
        try:
            if import_module(module_path).run() != 0:
                failed.append(name)
        except Exception as exc:
            print(f"  SUITE ERROR: {type(exc).__name__}: {exc}")
            failed.append(name)
        print()

    print("=" * 62)
    if failed:
        print(f"FAILED suites: {', '.join(failed)}")
        return 1
    print(f"All {len(SUITES)} suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
