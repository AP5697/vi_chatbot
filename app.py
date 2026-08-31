"""
app.py  --  VI Telecom Sales & Customer-Care Copilot (Streamlit UI)
==================================================================

A workbench for VI care/sales reps, in three tabs:

  Chat         free-form questions, answered by either the rule engine
               (offline) or Gemini (grounded on the full catalogue)
  Compare      pick 2-4 plans, get a side-by-side table -- no typing
  Plan Finder  answer a short questionnaire, get the cheapest plans that fit

Compare and Plan Finder read the dataset directly, so they keep working with no
API key and no quota left -- the parts of the tool a rep needs mid-call never
depend on an external service.

Run:
    streamlit run app.py

Setup:
    pip install -r requirements.txt
    copy .env.example to .env  and put your Gemini API key in it
    (get a key at https://aistudio.google.com/apikey)
"""

from __future__ import annotations

import sys

import streamlit as st

# Load .env if python-dotenv is available (optional convenience).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import time

from chatbot import get_catalog, build_system_instruction, GeminiCopilot, get_api_key
from chatbot import analytics as al
from chatbot.export import (comparison_markdown, conversation_markdown,
                            suggested_filename)
from chatbot.llm import QuotaExhaustedError, ServiceUnavailableError
from chatbot.rules_engine import (_days_text, _rupees, explain_pick,
                                  find_best_plans, render_comparison,
                                  render_details)

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

# --- Plan Finder questionnaire ------------------------------------------------
# Each answer maps to a constraint that find_best_plans() resolves against the
# dataset. Kept here (not in the engine) because these are UI wording choices.
BUDGET_OPTIONS = {
    "No limit": None,
    "Under ₹200": 200,
    "Under ₹400": 400,
    "Under ₹700": 700,
    "Under ₹1,000": 1000,
}

DATA_OPTIONS = {
    "Mostly calling — data barely used": None,
    "Light — about 1 GB/day": 1.0,
    "Moderate — about 1.5 GB/day": 1.5,
    "Heavy — 2 GB/day": 2.0,
    "Very heavy — 2.5 GB/day or more": 2.5,
}

VALIDITY_OPTIONS = {
    "Doesn't matter": (None, None),
    "Monthly — up to 30 days": (None, 30),
    "Two-monthly — 50+ days": (50, None),
    "Quarterly — 84+ days": (84, None),
    "Yearly — 300+ days": (300, None),
}

NEED_OPTIONS = {
    "5G ready": "5g",
    "OTT / entertainment bundled": "ott",
    "Unlimited calling": "voice",
    "International roaming": "roaming",
    "Data rollover": "rollover",
}

TYPE_OPTIONS = {"Either": "", "Prepaid": "Prepaid", "Postpaid": "Postpaid"}


# --------------------------------------------------------------------------
# Loaders (cached so a rerun never re-reads the CSV or rebuilds the prompt)
# --------------------------------------------------------------------------

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


@st.cache_resource(show_spinner=False)
def load_plan_frame():
    """
    The raw DataFrame, shared by Compare and Plan Finder.

    Loaded independently of the answer engine so those two tabs behave
    identically in both modes -- they are dataset features, not model features.
    """
    from chatbot.catalog import load_dataframe
    return load_dataframe()


def plan_label(row) -> str:
    """A picker label that stays unique even when two plans share a name."""
    return (f"{row['Plan_Name']} · {_rupees(row['Plan_Rental'])} · "
            f"{row['Validity_Days']}d · {row['Plan_SOC_ID']}")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

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

        st.divider()
        messages = st.session_state.get("messages", [])
        model = getattr(copilot, "active_model", "")
        st.download_button(
            "⬇️ Export conversation",
            data=conversation_markdown(messages, mode=mode, model=model,
                                       plan_count=plan_count),
            file_name=suggested_filename(),
            mime="text/markdown",
            use_container_width=True,
            disabled=not messages,
            help="Save this chat as a Markdown file for your call notes or CRM."
                 if messages else "Ask something first — then you can export it.",
        )
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------------------------------
# Tab: Chat
# --------------------------------------------------------------------------

def render_chat_tab(copilot, mode: str):
    st.caption("Ask anything about the VI plans — you'll get a direct, "
               "to-the-point answer from the dataset.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested questions, only before the chat starts.
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about a VI plan…")
    if "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the plan catalogue…"):
            t0 = time.monotonic()
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
            finally:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
        st.markdown(answer)
        al.log_event(
            question=prompt,
            engine=mode,
            model=getattr(copilot, "active_model", mode),
            ms=elapsed_ms,
            answer=answer,
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------
# Tab: Compare
# --------------------------------------------------------------------------

def render_compare_tab(df):
    st.caption("Pick 2–4 plans for a side-by-side table. Rows where every plan "
               "is identical are hidden, so only the differences show.")

    labels = {plan_label(r): i for i, r in df.iterrows()}
    chosen = st.multiselect(
        "Plans to compare", options=list(labels), max_selections=4,
        placeholder="Search by plan name, price, or code…",
    )

    if len(chosen) < 2:
        st.info("Select at least two plans to see the comparison.")
        return

    rows = df.loc[[labels[c] for c in chosen]]
    table = render_comparison(rows)
    st.markdown(table)

    names = list(rows["Plan_Name"])
    st.download_button(
        "⬇️ Export comparison",
        data=comparison_markdown(table, names),
        file_name=suggested_filename(prefix="vi-comparison"),
        mime="text/markdown",
    )


# --------------------------------------------------------------------------
# Tab: Plan Finder
# --------------------------------------------------------------------------

def render_finder_tab(df):
    st.caption("Answer a few questions about the customer and get the cheapest "
               "plans that meet every requirement — straight from the dataset.")

    with st.form("plan_finder"):
        c1, c2 = st.columns(2)
        budget = c1.selectbox("Customer's budget", list(BUDGET_OPTIONS))
        data_need = c2.selectbox("How much data do they use?", list(DATA_OPTIONS))
        c3, c4 = st.columns(2)
        validity = c3.selectbox("How often do they want to recharge?",
                                list(VALIDITY_OPTIONS))
        conn_type = c4.selectbox("Connection type", list(TYPE_OPTIONS))
        needs = st.multiselect("Must-haves", list(NEED_OPTIONS),
                               placeholder="Optional — leave empty if none")
        submitted = st.form_submit_button("🎯 Find matching plans",
                                          use_container_width=True)

    # Persist the answers so the results survive the rerun that a download
    # button click causes.
    if submitted:
        min_days, max_days = VALIDITY_OPTIONS[validity]
        st.session_state.finder_prefs = {
            "max_price": BUDGET_OPTIONS[budget],
            "min_daily_gb": DATA_OPTIONS[data_need],
            "min_days": min_days,
            "max_days": max_days,
            "product_type": TYPE_OPTIONS[conn_type],
            "needs": [NEED_OPTIONS[n] for n in needs],
        }

    prefs = st.session_state.get("finder_prefs")
    if not prefs:
        return

    rows, relaxed = find_best_plans(prefs, df, k=3)

    if relaxed:
        st.warning("No plan met every requirement, so these were relaxed: "
                   + ", ".join(f"**{r}**" for r in relaxed))
    if rows.empty:
        st.error("No live plans in the catalogue match, even after relaxing "
                 "every requirement.")
        return

    st.success(f"Top {len(rows)} match{'es' if len(rows) > 1 else ''}, "
               f"cheapest first.")

    for col, (_, row) in zip(st.columns(len(rows)), rows.iterrows()):
        with col, st.container(border=True):
            st.markdown(f"**{row['Plan_Name']}**")
            st.metric("Price", _rupees(row["Plan_Rental"]),
                      _days_text(row["Validity_Days"]), delta_color="off")
            for reason in explain_pick(row, prefs):
                st.caption(f"• {reason}")
            with st.expander("Full details"):
                st.markdown(render_details(row))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
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

    st.title("📶 VI Telecom Sales & Customer-Care Copilot")

    if mode == LLM_MODE and not has_key:
        st.error("No Gemini API key found — switch to **Rule-based (no LLM)** in the "
                 "sidebar to use the app without a key.")
        st.markdown(
            "To use Gemini mode, set **`GEMINI_API_KEY`**:\n\n"
            "- **Locally:** copy `.env.example` to `.env` and paste your key\n"
            "- **Streamlit Cloud:** app Settings → Secrets → `GEMINI_API_KEY = \"...\"`\n\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
        st.stop()

    try:
        if mode == RULES_MODE:
            copilot, facts, plan_count = load_rules_engine()
        else:
            copilot, facts, plan_count = load_gemini_copilot()
        df = load_plan_frame()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python build_wide_dataset.py` to generate the dataset, then reload.")
        st.stop()
        return

    render_sidebar(copilot, facts, plan_count, mode)

    tab_chat, tab_compare, tab_finder = st.tabs(
        ["💬 Chat", "⚖️ Compare plans", "🎯 Plan Finder"]
    )
    with tab_chat:
        render_chat_tab(copilot, mode)
    with tab_compare:
        render_compare_tab(df)
    with tab_finder:
        render_finder_tab(df)


if __name__ == "__main__":
    main()
