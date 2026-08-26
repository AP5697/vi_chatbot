"""
cli.py
======

A standalone, zero-LLM command-line interface to the VI plan dataset.

No Streamlit, no browser, no API key, no network call -- just the rule-based
engine (chatbot/rules_engine.py) reading output/vi_plans_wide_dataset.csv and
answering directly in the terminal. Useful for quick lookups, demos, or
scripting against the dataset without spinning up the web app.

Usage
-----
Interactive chat (keeps context across turns, like the Streamlit chat does):
    python cli.py

One-shot: ask a single question and exit (handy for demos/scripts):
    python cli.py "What is the price of the Vi Hero Unlimited 449 plan?"

Exit the interactive loop with: exit / quit / Ctrl+C
"""

from __future__ import annotations

import sys

from chatbot.catalog import load_dataframe, dataset_facts
from chatbot.rules_engine import RulesEngine

BANNER = """\
============================================================
 VI Plan Assistant -- Rule-Based CLI (no LLM, fully offline)
============================================================"""


def load_engine() -> tuple[RulesEngine, dict]:
    df = load_dataframe()
    return RulesEngine(df), dataset_facts(df)


def print_intro(facts: dict) -> None:
    print(BANNER)
    print(f" Plans loaded : {facts['plan_count']}")
    if facts.get("min_price") is not None:
        print(f" Price range  : Rs.{facts['min_price']} - Rs.{facts['max_price']}")
    print(f" Plan types   : {', '.join(facts.get('plan_types', []))}")
    print("-" * 60)
    print(" Ask anything about the VI plans. Type 'exit' to quit.")
    print(" Examples: 'price of plan 26106058' | 'plans under 500 with 5G'")
    print("           'compare 299 and 449'    | 'cheapest plan with 2GB/day'")
    print(BANNER)
    print()


def run_one_shot(engine: RulesEngine, question: str) -> None:
    answer = engine.reply([{"role": "user", "content": question}])
    print(answer)


def run_interactive(engine: RulesEngine) -> None:
    history: list[dict] = []
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye"):
            print("Goodbye.")
            return

        history.append({"role": "user", "content": question})
        answer = engine.reply(history)
        history.append({"role": "assistant", "content": answer})

        print(f"\nBot> {answer}\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        engine, facts = load_engine()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Run  python build_wide_dataset.py  first to generate the dataset.",
              file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    if args:
        run_one_shot(engine, " ".join(args))
    else:
        print_intro(facts)
        run_interactive(engine)


if __name__ == "__main__":
    main()
