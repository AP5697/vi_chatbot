"""
test_rules.py
=============

Regression tests for the rule-based engine (rules_engine.py).

Unlike chatbot/eval.py (which calls the live Gemini API), this runs entirely
offline and deterministically -- the rule engine has no randomness or network,
so the same question always yields the same answer. That makes these safe to
run on every change to catch regressions instantly.

Run:  python -m chatbot.test_rules
"""

from __future__ import annotations

import re
import sys

from .catalog import load_dataframe
from .rules_engine import RulesEngine


class Case:
    """One test: ask `q`, assert the answer satisfies every check."""

    def __init__(self, q, contains=None, excludes=None, regex=None,
                 history=None, label=""):
        self.q = q
        self.contains = contains or []
        self.excludes = excludes or []
        self.regex = regex
        self.history = history
        self.label = label or q


def _run_case(engine: RulesEngine, case: Case) -> tuple[bool, str]:
    history = case.history or []
    history = history + [{"role": "user", "content": case.q}]
    answer = engine.reply(history)
    low = answer.lower()

    for needle in case.contains:
        if needle.lower() not in low:
            return False, f"missing {needle!r}\n    got: {answer[:200]}"
    for needle in case.excludes:
        if needle.lower() in low:
            return False, f"should NOT contain {needle!r}\n    got: {answer[:200]}"
    if case.regex and not re.search(case.regex, answer, re.I):
        return False, f"regex {case.regex!r} not found\n    got: {answer[:200]}"
    return True, "ok"


def build_cases(df) -> list[Case]:
    # Anchor a few tests on real values pulled from the dataset so they stay
    # valid if the data changes.
    p449 = df[df["Plan_Name"].str.contains("449", na=False)].iloc[0]
    pid = p449["Plan_SOC_ID"]

    return [
        # --- single-field lookups ---
        Case(f"What is the price of plan {pid}?", contains=[pid, "₹449"],
             label="price lookup by code"),
        Case(f"does plan {pid} have 5G?", contains=[pid], regex=r"5G.*:\s*No",
             label="5G yes/no lookup"),
        Case(f"how many days validity does plan {pid} have?",
             contains=["56 days"], excludes=["62 plans"],
             label="validity is a field, not a count"),

        # --- full details ---
        Case(f"complete details of plan {pid}",
             contains=["Vi Hero Unlimited 449", "Price", "Validity", "Voice"],
             label="full details card"),

        # --- filtered search ---
        Case("plans under 500 with unlimited voice",
             contains=["plans match", "price <= ₹500"],
             regex=r"\*\*2[0-9]\*\* plans",  # ~24 plans, not the 2-plan bug
             label="price + unlimited voice (no plan-type leak)"),
        Case("5G plans under 400", contains=["price <= ₹400", "5g eligible = yes"],
             label="5G + price filter"),
        Case("plans between 200 and 400",
             contains=["price >= ₹200", "price <= ₹400"],
             label="price range"),

        # --- OTT ---
        Case("which plans include Netflix", contains=["ott includes netflix"],
             label="OTT brand search"),
        Case("how many prepaid plans have OTT",
             regex=r"\*\*\d+\*\* plans", contains=["prepaid", "ott"],
             label="count with filters"),

        # --- superlatives ---
        Case("cheapest plan with 2GB/day",
             contains=["daily data >= 2.0gb", "sorted by lowest price"],
             label="cheapest + data filter"),
        Case("which plan has the most data", contains=["highest daily data"],
             label="most data superlative"),
        Case("longest validity plan", contains=["longest validity"],
             label="longest validity superlative"),

        # --- comparison ---
        Case("compare 249 and 299", contains=["| Feature |", "Vi Hero Unlimited 249"],
             label="comparison table"),

        # --- alternatives ---
        Case(f"alternatives to plan {pid}",
             contains=["alternatives to", "Vi Hero Unlimited 449"],
             excludes=["Data Only Pack"],  # must not offer a voiceless booster
             label="alternatives are same voice class"),
        Case(f"similar to the 449 plan", contains=["alternatives to"],
             label="'similar to' triggers alternatives"),

        # --- recommendations by profile ---
        Case("recommend a plan for a heavy data user",
             contains=["most daily data"], label="reco: heavy data"),
        Case("suggest something for a budget student",
             contains=["lowest price"], label="reco: budget"),
        Case("best plan for a traveller", contains=["roaming"],
             label="reco: traveller"),
        Case("recommend a plan", contains=["customer's priority"],
             label="reco with no profile -> asks for priority"),

        # --- follow-up context ---
        Case("what is its validity?", contains=["56 days"],
             history=[{"role": "user", "content": f"details of plan {pid}"},
                      {"role": "assistant", "content": "..."}],
             label="follow-up reuses prior plan"),
        Case("compare the first two", contains=["| Feature |", "Vi Hero Unlimited 99"],
             history=[{"role": "user", "content": "plans under 200"},
                      {"role": "assistant", "content": "..."}],
             label="positional follow-up: first two"),
        Case("details of the second one", contains=["Vi Hero Unlimited 149"],
             history=[{"role": "user", "content": "plans under 200"},
                      {"role": "assistant", "content": "..."}],
             label="positional follow-up: second one"),

        # --- guardrails ---
        Case("complete details of plan 99999999",
             contains=["no plan with code 99999999"],
             label="unknown plan code handled"),
        Case("what should I pitch to someone who likes hiking",
             contains=["couldn't turn that into a lookup"],
             label="open-ended -> honest fallback"),
    ]


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_dataframe()
    engine = RulesEngine(df)
    cases = build_cases(df)

    passed = 0
    print(f"Running {len(cases)} rule-engine regression tests...\n")
    for i, case in enumerate(cases, 1):
        ok, reason = _run_case(engine, case)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{i:>2}/{len(cases)}] {status}  {case.label}")
        if not ok:
            print(f"        {reason}")

    print(f"\nRESULT: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(run())
