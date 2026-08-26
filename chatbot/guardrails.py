"""
guardrails.py
=============

Lightweight, code-only safety checks for the Gemini answer path. No NeMo
Guardrails, no LangChain, and -- deliberately -- NO extra LLM calls: every
check here is plain pattern-matching or a dataset lookup, so it adds zero API
cost and cannot itself be rate-limited. (An LLM-based guardrail would fire a
second Gemini request per question, which directly fights the free-tier quota
this project already had to engineer around.)

Three guards:

  1. INPUT  -- jailbreak / prompt-injection detection.
     Runs BEFORE the API call, so a manipulation attempt is refused without
     spending quota and without the model ever seeing it.

  2. TOPIC  -- staying on VI plans.
     Enforced primarily in the system prompt ("say so if unrelated"). We do NOT
     hard-block here, because a keyword topic filter produces false positives on
     legitimate plan questions. Documented as a design choice.

  3. OUTPUT -- fact verification against the dataset.
     After the model answers, we check that every plan code it cites actually
     exists. Grounding already makes invention unlikely; this is the backstop
     that catches it deterministically if it ever happens.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

# Patterns that signal an attempt to override instructions, extract the system
# prompt, or make the assistant role-play as something else. Matched case-
# insensitively against the raw user text.
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+|your\s+|the\s+|any\s+)?(previous\s+|above\s+|prior\s+)?(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|the\s+|your\s+|any\s+)?(above|previous|prior|system|earlier)",
    r"forget\s+(everything|all|your\s+instructions?|the\s+above|what\s+you)",
    r"you\s+are\s+now\b",
    r"pretend\s+(to\s+be|you\s+are|that\s+you)",
    r"\bact\s+as\s+(if\s+)?(an?\s+|though)",
    r"role[-\s]?play\s+as",
    r"developer\s+mode",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"(reveal|show|print|repeat|tell\s+me|what\s+(is|are))\s+(your\s+|the\s+)?(system\s+)?(prompt|instructions?)",
    r"(words|text|everything)\s+above",
    r"new\s+instructions?\s*:",
    r"\boverride\s+(your|the|all|system)",
    r"bypass\s+(your|the|all|safety|guard)",
]

_COMPILED = [re.compile(p, re.I) for p in JAILBREAK_PATTERNS]

INJECTION_REFUSAL = (
    "I can only help with questions about the VI plan dataset. I can't change my "
    "instructions, reveal my configuration, or take on a different role."
)


def check_input(question: str) -> str | None:
    """
    Return a refusal message if the question looks like a jailbreak / prompt-
    injection attempt, else None. Cheap: pure regex, runs before any API call.
    """
    for pattern in _COMPILED:
        if pattern.search(question or ""):
            return INJECTION_REFUSAL
    return None


def verify_output(answer: str, valid_ids: Iterable[str]) -> str:
    """
    Backstop fact-check: confirm every plan code the answer cites exists in the
    dataset. If the model invented a code, append a clear warning rather than
    letting a fabricated identifier pass silently.

    Only 8-digit SOC-ID-shaped tokens are checked, so ordinary numbers (prices,
    validity days, GB) are ignored. Returns the answer unchanged when clean.
    """
    if not answer:
        return answer

    real_ids = set(map(str, valid_ids))
    cited = set(re.findall(r"\b\d{8}\b", answer))
    invented = [c for c in cited if c not in real_ids]

    if invented:
        codes = ", ".join(sorted(invented))
        answer += (
            f"\n\n---\n⚠️ *Note: plan code(s) {codes} are not in the VI dataset. "
            f"Please verify — this may be an error.*"
        )
    return answer


def scrub(question: str, answer_fn: Callable[[], str], valid_ids: Iterable[str]) -> str:
    """
    Apply both guards around an answer function.

    answer_fn() is called ONLY if the input passes the injection check, so a
    blocked prompt never reaches the model (saving a request). Its result is
    then fact-verified before being returned.
    """
    refusal = check_input(question)
    if refusal:
        return refusal
    return verify_output(answer_fn(), valid_ids)
