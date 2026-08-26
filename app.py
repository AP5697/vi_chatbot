"""
app.py  --  VI Telecom Sales & Customer-Care Copilot (Streamlit UI)
==================================================================

A chat interface for VI care/sales reps. It loads the segregated plan
workbook, grounds Google Gemini on the full catalogue, and answers plan
questions in sales-ready Markdown.

Run:
    streamlit run app.py

Setup:
    pip install -r requirements.txt
    copy .env.example to .env  and put your Gemini API key in it
    (get a key at https://aistudio.google.com/apikey)
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Load .env if python-dotenv is available (optional convenience).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from chatbot import get_catalog, build_system_instruction, GeminiCopilot, get_api_key
from chatbot.llm import QuotaExhaustedError, ServiceUnavailableError

st.set_page_config(page_title="VI Sales Copilot", page_icon="📶", layout="wide")

# Windows consoles choke on the rupee symbol; make stdout UTF-8 defensively.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUGGESTED_QUESTIONS = [
    "What's the price of the Vi Hero Unlimited 449 plan?",
    "Which plans have 5G and cost under ₹400?",
    "Does the ₹299 plan include any OTT?",
    "How many days validity does plan 26106058 have?",
    "Compare the ₹299 and ₹449 plans.",
    "Which is the cheapest plan with 2GB/day?",
]


RULES_MODE = "Rule-based (no LLM)"
LLM_MODE = "Gemini-powered"


@st.cache_resource(show_spinner="Loading VI plan catalogue…")
def load_gemini_copilot():
    """Dataset + grounded Gemini client (with guardrails). Needs an API key."""
    records, grounding, facts = get_catalog()
    system_instruction = build_system_instruction(grounding, facts)
    valid_ids = {str(r.get("Plan_SOC_ID")) for r in records}
    copilot = GeminiCopilot(system_instruction=system_instruction,
                            valid_plan_ids=valid_ids)
    return copilot, facts, len(records)


@st.cache_resource(show_spinner="Loading VI plan catalogue…")
def load_rules_engine():
    """Dataset + deterministic engine. No API key, no network, no quota."""
    from chatbot.catalog import load_dataframe, dataset_facts
    from chatbot.rules_engine import RulesEngine

    df = load_dataframe()
    return RulesEngine(df), dataset_facts(df), len(df)


def render_sidebar(copilot, facts: dict, plan_count: int, mode: str):
    with st.sidebar:
        st.caption("Grounded on the authoritative VI plan dataset. "
                   "Answers come only from the loaded catalogue — no guessing.")
        st.divider()
        st.metric("Plans loaded", plan_count)
        if facts.get("min_price") is not None:
            st.metric("Price range", f"₹{facts['min_price']} – ₹{facts['max_price']}")
        st.caption("**Plan types:** " + ", ".join(facts.get("plan_types", [])))
        st.divider()

        if mode == RULES_MODE:
            st.caption("Engine: `rule-based` — offline, no API key, no quota.")
            st.caption(
                "Answers come from direct lookups on the dataset, so nothing can "
                "be invented. Free-form questions may need concrete criteria "
                "(a plan code/name, or a price / data / validity / 5G / OTT filter)."
            )
        else:
            st.caption(f"Model: `{copilot.active_model}`")
            with st.expander("Fallback chain"):
                st.caption(
                    "Free-tier quota is counted per model per day, so if one runs "
                    "out the app automatically moves to the next:"
                )
                for i, m in enumerate(copilot.model_chain, start=1):
                    st.caption(f"{i}. `{m}`")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    # --- Engine mode --------------------------------------------------------
    # Default to rule-based when there's no key, so the app is always usable.
    has_key = bool(get_api_key())
    if "mode" not in st.session_state:
        st.session_state.mode = LLM_MODE if has_key else RULES_MODE

    with st.sidebar:
        st.header("📶 VI Sales Copilot")
        st.radio(
            "Answer engine",
            [RULES_MODE, LLM_MODE],
            key="mode",
            help="Rule-based runs entirely offline with no API key or quota. "
                 "Gemini handles free-form phrasing but needs a key.",
        )
    mode = st.session_state.mode

    if mode == LLM_MODE and not has_key:
        st.title("📶 VI Telecom Sales Copilot")
        st.error("No Gemini API key found — switch to **Rule-based (no LLM)** in the "
                 "sidebar to use the app without a key.")
        st.markdown(
            "To use Gemini mode, set **`GEMINI_API_KEY`**:\n\n"
            "- **Locally:** copy `.env.example` to `.env` and paste your key\n"
            "- **Streamlit Cloud:** app Settings → Secrets → `GEMINI_API_KEY = \"...\"`\n\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        st.stop()

    # --- Load dataset + engine ----------------------------------------------
    try:
        if mode == RULES_MODE:
            copilot, facts, plan_count = load_rules_engine()
        else:
            copilot, facts, plan_count = load_gemini_copilot()
    except FileNotFoundError as exc:
        st.title("📶 VI Telecom Sales Copilot")
        st.error(str(exc))
        st.info("Run `python build_wide_dataset.py` to generate the dataset, then reload.")
        st.stop()
        return

    render_sidebar(copilot, facts, plan_count, mode)

    st.title("📶 VI Telecom Sales & Customer-Care Copilot")
    st.caption("Ask anything about the VI plans — you'll get a direct, to-the-point answer from the dataset.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Suggested questions (only before the chat starts) -----------------
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

    # --- Replay history -----------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Input (typed or from a suggestion button) --------------------------
    prompt = st.chat_input("Ask about a VI plan…")
    if "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking the plan catalogue…"):
                try:
                    answer = copilot.reply(st.session_state.messages)
                except QuotaExhaustedError:
                    # Every model's free daily allowance is spent. Say what
                    # happened and what to do -- no raw traceback.
                    answer = (
                        "**Daily free-tier quota is used up.**\n\n"
                        "Google's free Gemini tier allows only a limited number of "
                        "requests per model per day, and every model in the fallback "
                        "chain is now spent.\n\n"
                        "You can:\n"
                        "- switch to **Rule-based (no LLM)** in the sidebar — no quota at all, or\n"
                        "- wait for the quota to reset (it resets daily), or\n"
                        "- enable billing on your key at https://aistudio.google.com/apikey\n\n"
                        "Nothing is wrong with the app or the dataset."
                    )
                except ServiceUnavailableError:
                    # Google-side overload, already retried across every model.
                    answer = (
                        "**Gemini is busy right now.**\n\n"
                        "Every model was retried and all reported high demand "
                        "(HTTP 503). This is temporary and on Google's side — "
                        "please send your question again in a moment.\n\n"
                        "Nothing is wrong with the app or the dataset."
                    )
                except Exception as exc:  # surface real errors cleanly
                    answer = f"⚠️ Something went wrong talking to the model:\n\n`{exc}`"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
