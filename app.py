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


@st.cache_resource(show_spinner="Loading VI plan catalogue…")
def load_copilot():
    """Load the dataset once and build the grounded Gemini client (cached)."""
    records, grounding, facts = get_catalog()
    system_instruction = build_system_instruction(grounding, facts)
    copilot = GeminiCopilot(system_instruction=system_instruction)
    return copilot, facts, len(records)


def render_sidebar(copilot, facts: dict, plan_count: int):
    with st.sidebar:
        st.header("📶 VI Sales Copilot")
        st.caption("Grounded on the authoritative VI plan dataset. "
                   "Answers come only from the loaded catalogue — no guessing.")
        st.divider()
        st.metric("Plans loaded", plan_count)
        if facts.get("min_price") is not None:
            st.metric("Price range", f"₹{facts['min_price']} – ₹{facts['max_price']}")
        st.caption("**Plan types:** " + ", ".join(facts.get("plan_types", [])))
        st.divider()
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
    # --- API key gate -------------------------------------------------------
    if not get_api_key():
        st.title("📶 VI Telecom Sales Copilot")
        st.error("No Gemini API key found.")
        st.markdown(
            "Set **`GEMINI_API_KEY`** (or `GOOGLE_API_KEY`) before launching.\n\n"
            "1. Get a free key at https://aistudio.google.com/apikey\n"
            "2. Copy `.env.example` to `.env` and paste your key, **or** set it in your shell:\n"
            "   ```bash\n   set GEMINI_API_KEY='your_key_here'   # Windows (cmd)\n   ```\n"
            "3. Re-run `streamlit run app.py`."
        )
        st.stop()

    # --- Load dataset + model ----------------------------------------------
    try:
        copilot, facts, plan_count = load_copilot()
    except FileNotFoundError as exc:
        st.title("📶 VI Telecom Sales Copilot")
        st.error(str(exc))
        st.info("Run `python segregate_telecom_data.py` to generate the workbook, then reload.")
        st.stop()
        return

    render_sidebar(copilot, facts, plan_count)

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
