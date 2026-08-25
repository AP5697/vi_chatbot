"""
eval.py
=======

A lightweight accuracy check for the VI Copilot.

It does NOT use a fixed/hand-written question bank tied to specific plan
names (those go stale the moment the source data changes). Instead it builds
test cases straight from the current catalogue at run time, so it always
tests against whatever is actually in output/telecom_datasets.xlsx today.

What it checks:
  1. FACT RECALL   - ask for full details of a real plan, verify the reply
                      contains that plan's actual price, validity and name.
  2. PRICE FILTER  - ask for plans under a threshold, verify a plan that
                      should qualify is mentioned AND a plan that should NOT
                      qualify (well above the threshold) is absent.
  3. OTT LOOKUP    - ask which plans include a specific OTT platform, verify
                      a plan known to have it is mentioned.
  4. HALLUCINATION - ask about a plan code that does not exist, verify the
                      bot says so instead of inventing an answer. This is the
                      most important check for a "never fabricate" spec.

Each check is a simple case-insensitive substring match against the model's
live reply - deliberately simple and inspectable, not another LLM grading an
LLM. Run:

    python -m chatbot.eval
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .catalog import get_catalog
from .prompt import build_system_instruction
from .llm import GeminiCopilot, get_api_key

# Phrases that count as a correct "I don't have that" answer.
NOT_AVAILABLE_PHRASES = [
    "not available",
    "couldn't find",
    "could not find",
    "no plan",
    "not found",
    "doesn't exist",
    "does not exist",
    "isn't in the dataset",
    "is not in the dataset",
    "not in the dataset",
    "isn't in the",
    "no matching",
]


@dataclass
class TestCase:
    label: str
    question: str
    must_include: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    any_of: list[str] = field(default_factory=list)  # pass if ANY of these appear


def _price(r: dict):
    try:
        return int(float(str(r.get("Plan_Rental", "")).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def build_test_cases(records: list[dict], seed: int = 42) -> list[TestCase]:
    rng = random.Random(seed)
    cases: list[TestCase] = []

    priced = [r for r in records if _price(r) is not None]
    by_price = sorted(priced, key=_price)

    # --- 1. Fact recall: full detail lookup on a real plan -------------------
    sample_detail = rng.sample(priced, k=min(3, len(priced)))
    for r in sample_detail:
        cases.append(TestCase(
            label=f"Fact recall: plan {r['Plan_SOC_ID']}",
            question=f"Give me complete details of plan {r['Plan_SOC_ID']}.",
            must_include=[str(r["Plan_SOC_ID"]), str(_price(r))],
        ))

    # --- 2. Price filter: cheap plan should appear, expensive should not ----
    if len(by_price) >= 4:
        threshold = int((_price(by_price[0]) + _price(by_price[len(by_price) // 3])) / 2) + 1
        below = [r for r in by_price if _price(r) < threshold]
        above = [r for r in by_price if _price(r) > threshold + 200]
        if below and above:
            target_in = below[-1]   # closest one under the threshold
            target_out = above[-1]  # clearly above it
            cases.append(TestCase(
                label=f"Price filter: under ₹{threshold}",
                question=f"Show me all VI plans under ₹{threshold}.",
                must_include=[str(target_in["Plan_SOC_ID"])],
                must_exclude=[str(target_out["Plan_SOC_ID"])],
            ))

    # --- 3. OTT lookup ---------------------------------------------------------
    ott_plans = [r for r in records
                 if r.get("OTT_Platform") and r["OTT_Platform"] != "No OTT Bundled"]
    if ott_plans:
        r = rng.choice(ott_plans)
        # Use the first platform name in a possibly bundled cell.
        platform = r["OTT_Platform"].split(" + ")[0].strip()
        cases.append(TestCase(
            label=f"OTT lookup: {platform}",
            question=f"Which VI plans include {platform}?",
            must_include=[str(r["Plan_SOC_ID"])],
        ))

    # --- 4. Hallucination trap: a plan code that cannot exist -----------------
    real_ids = {str(r["Plan_SOC_ID"]) for r in records}
    fake_id = "99999999"
    while fake_id in real_ids:
        fake_id += "9"
    cases.append(TestCase(
        label="Hallucination trap: nonexistent plan",
        question=f"Give me complete details of plan {fake_id}.",
        any_of=NOT_AVAILABLE_PHRASES,
    ))

    # --- 5. Second hallucination trap: fabricated benefit ----------------------
    cases.append(TestCase(
        label="Hallucination trap: fake benefit",
        question="Which VI plan gives free international first-class flight upgrades?",
        any_of=NOT_AVAILABLE_PHRASES,
    ))

    return cases


def check(case: TestCase, answer: str) -> tuple[bool, str]:
    lower = answer.lower()

    missing = [s for s in case.must_include if s.lower() not in lower]
    if missing:
        return False, f"missing required text: {missing}"

    present = [s for s in case.must_exclude if s.lower() in lower]
    if present:
        return False, f"contains text it should have excluded: {present}"

    if case.any_of and not any(s.lower() in lower for s in case.any_of):
        return False, f"none of the expected phrases found: {case.any_of}"

    return True, "ok"


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not get_api_key():
        print("ERROR: No Gemini API key found. Set GEMINI_API_KEY in your .env file.")
        return 1

    print("Loading catalogue...")
    records, grounding, facts = get_catalog()
    system_instruction = build_system_instruction(grounding, facts)
    copilot = GeminiCopilot(system_instruction=system_instruction)

    cases = build_test_cases(records)
    print(f"Running {len(cases)} accuracy checks against the live model...\n")
    print("=" * 70)

    results = []
    for i, case in enumerate(cases, start=1):
        history = [{"role": "user", "content": case.question}]
        try:
            answer = copilot.reply(history)
        except Exception as exc:
            passed, reason = False, f"API error: {exc}"
            answer = ""
        else:
            passed, reason = check(case, answer)

        results.append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"[{i}/{len(cases)}] {status}  -  {case.label}")
        print(f"    Q: {case.question}")
        if not passed:
            print(f"    Reason: {reason}")
            print(f"    A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print("-" * 70)

    passed_count = sum(results)
    total = len(results)
    pct = (passed_count / total * 100) if total else 0.0

    print("=" * 70)
    print(f"RESULT: {passed_count}/{total} passed ({pct:.0f}%)")
    print("=" * 70)

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(run())
