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
from .rules_engine import (RulesEngine, _daily_gb, _is_negative, _num,
                           _table_cell, explain_pick, find_best_plans,
                           render_comparison, render_table)


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
        # The CSV stores Total_Data_Limit unitless ("84"), which reads as days
        # or rupees unless the renderer adds GB.
        Case(f"total data on plan {pid}", contains=["84 GB"],
             label="total data carries its GB unit"),

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


# --------------------------------------------------------------------------
# Plan Finder (find_best_plans) -- a function API rather than a question, so it
# gets its own small harness: each case is (label, prefs, check) where check
# returns None on success or a reason string on failure.
# --------------------------------------------------------------------------

def build_finder_cases() -> list[tuple[str, dict, object]]:
    def all_rows(rows, predicate, what):
        bad = [r["Plan_Name"] for _, r in rows.iterrows() if not predicate(r)]
        return f"{what} violated by: {bad}" if bad else None

    return [
        ("budget ≤₹200 + unlimited voice",
         {"max_price": 200, "needs": ["voice"]},
         lambda rows, relaxed: (
             "relaxed unexpectedly: " + str(relaxed) if relaxed else
             all_rows(rows, lambda r: (_num(r["Plan_Rental"]) or 0) <= 200
                      and "unlimited" in str(r["Voice_Benefit"]).lower(),
                      "budget/voice"))),

        ("heavy data + 5G + OTT",
         {"max_price": 700, "min_daily_gb": 2.0, "needs": ["5g", "ott", "voice"]},
         lambda rows, relaxed: (
             "relaxed unexpectedly: " + str(relaxed) if relaxed else
             all_rows(rows, lambda r: str(r["5G_Eligible"]).lower() == "yes"
                      and (_daily_gb(r["Daily_Data_Limit"]) or 0) >= 2.0
                      and not _is_negative(r["OTT_Platform"]),
                      "5G/data/OTT"))),

        ("quarterly validity (84+ days)",
         {"min_days": 84, "needs": ["voice"]},
         lambda rows, relaxed: all_rows(
             rows, lambda r: (_num(r["Validity_Days"]) or 0) >= 84, "min validity")),

        ("monthly validity (≤30 days)",
         {"max_days": 30, "needs": ["voice"]},
         lambda rows, relaxed: all_rows(
             rows, lambda r: (_num(r["Validity_Days"]) or 0) <= 30, "max validity")),

        ("postpaid only",
         {"product_type": "Postpaid"},
         lambda rows, relaxed: all_rows(
             rows, lambda r: r["Product_Type"] == "Postpaid", "product type")),

        ("only live plans are ever offered",
         {"max_price": 1000},
         lambda rows, relaxed: all_rows(
             rows, lambda r: str(r["Plan_Status"]).lower() == "live", "live status")),

        ("cheapest first",
         {"needs": ["voice"]},
         lambda rows, relaxed: (
             None if list(p := [_num(r["Plan_Rental"]) for _, r in rows.iterrows()])
             == sorted(p) else f"not price-ascending: {p}")),

        # The rep must never be handed an empty result with no explanation.
        ("impossible ask still returns picks + explains what gave",
         {"max_price": 50, "min_daily_gb": 3.0,
          "needs": ["5g", "ott", "roaming", "rollover"]},
         lambda rows, relaxed: (
             "returned nothing" if rows.empty else
             "nothing reported as relaxed" if not relaxed else None)),

        # Budget is the customer's hardest constraint, so it must be the LAST
        # thing dropped -- never sacrificed while softer asks still stand.
        ("budget is relaxed last",
         {"max_price": 50, "min_daily_gb": 3.0, "needs": ["5g", "ott"]},
         lambda rows, relaxed: (
             None if not any("budget" in r for r in relaxed)
             or relaxed[-1].startswith("budget")
             else f"budget dropped too early: {relaxed}")),

        ("explain_pick cites the satisfied must-haves",
         {"max_price": 700, "needs": ["5g", "voice"]},
         lambda rows, relaxed: (
             "no rows" if rows.empty else
             None if any("5G" in x for x in explain_pick(rows.iloc[0],
                                                         {"needs": ["5g", "voice"]}))
             else "5G not mentioned in reasons")),
    ]


# --------------------------------------------------------------------------
# Markdown table integrity
#
# Nine dataset columns legitimately contain a '|' separator. Unescaped, that
# opens a phantom column and silently shifts every later value -- so the rep
# reads one plan's benefit under another plan's name. These check the STRUCTURE
# (equal cell counts per row) rather than any particular wording.
# --------------------------------------------------------------------------

def _row_widths(table: str) -> list[int]:
    """Cell count of each row, treating an escaped \\| as ordinary text."""
    widths = []
    for line in table.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        widths.append(len(line.replace("\\|", "\0").strip("|").split("|")))
    return widths


def run_table_tests(df) -> tuple[int, int]:
    engine = RulesEngine(df)

    # A plan whose Additional_Benefits really does contain a pipe.
    piped = df[df["Additional_Benefits"].str.contains(r"\|", na=False, regex=True)]
    two = piped.head(2)

    cases: list[tuple[str, object]] = [
        ("comparison of pipe-containing plans keeps its shape",
         lambda: render_comparison(two)),
        ("plan list keeps its shape", lambda: render_table(df.head(8))),
        ("comparison of 4 plans keeps its shape",
         lambda: render_comparison(df.head(4))),
        ("engine answer for a filtered search keeps its shape",
         lambda: engine.reply([{"role": "user", "content": "5G plans under 400"}])),
    ]

    passed = 0
    print(f"\nMarkdown table integrity ({len(cases) + 1} tests):")
    for label, produce in cases:
        widths = _row_widths(produce())
        ok = len(set(widths)) <= 1
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"         ragged rows: {widths}")

    # Escaping must PRESERVE the value, not truncate it at the pipe. Checked on
    # the escaping unit directly: both segments of a piped benefit must survive,
    # separated by an ESCAPED pipe (so it renders as text, not a new column).
    raw = "Bundled OTT Subscriptions | Vi Movies & TV Access"
    escaped = _table_cell("Additional_Benefits", raw)
    ok = (escaped == "Bundled OTT Subscriptions \\| Vi Movies & TV Access")
    passed += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] escaped pipe preserves the full value")
    if not ok:
        print(f"         got {escaped!r}")
    return passed, len(cases) + 1


def run_finder_tests(df) -> tuple[int, int]:
    cases = build_finder_cases()
    passed = 0
    print(f"\nPlan Finder ({len(cases)} tests):")
    for label, prefs, check in cases:
        rows, relaxed = find_best_plans(prefs, df)
        try:
            reason = check(rows, relaxed)
        except Exception as exc:               # a raising check is a failure
            reason = f"raised {type(exc).__name__}: {exc}"
        passed += reason is None
        print(f"  [{'PASS' if reason is None else 'FAIL'}] {label}")
        if reason:
            print(f"         {reason}")
    return passed, len(cases)


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

    total = len(cases)
    for suite in (run_table_tests, run_finder_tests):
        suite_passed, suite_total = suite(df)
        passed += suite_passed
        total += suite_total

    print(f"\nRESULT: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
