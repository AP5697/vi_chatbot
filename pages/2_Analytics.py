"""
pages/2_Analytics.py  --  Usage analytics (Streamlit multipage)
================================================================

Shows what reps actually ask, which engine they use, which plans come up most,
and how activity changes over time. All data comes from output/analytics.jsonl
which is written silently every time a question is answered in the Chat tab.

The page is passcode-protected with the same ADMIN_PASSCODE as the admin panel,
so both management pages share one credential. When no passcode is set the page
stays open (with a warning), so local demos work without any setup.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from chatbot import analytics as al
from chatbot.llm import _get_setting

st.set_page_config(page_title="VI Copilot · Analytics", page_icon="📊", layout="wide")
st.title("📊 Usage Analytics")
st.caption("What reps are asking, which engine they use, and which plans come up most.")


# --------------------------------------------------------------------------
# Passcode gate — same credential as the admin panel
# --------------------------------------------------------------------------

def gate() -> bool:
    passcode = (_get_setting("ADMIN_PASSCODE") or "").strip()
    if not passcode:
        st.warning(
            "**No `ADMIN_PASSCODE` set.** This page is visible to anyone. "
            "Set it in `.env` (local) or Streamlit Cloud Secrets to lock it."
        )
        return True
    if st.session_state.get("admin_ok"):
        return True
    st.info("This page is passcode-protected.")
    entered = st.text_input("Admin passcode", type="password")
    if entered and entered == passcode:
        st.session_state.admin_ok = True
        st.rerun()
    elif entered:
        st.error("Incorrect passcode.")
    return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if not gate():
    st.stop()

events = al.load_events()

if not events:
    st.info(
        "No questions logged yet. Ask something in the **💬 Chat** tab and "
        "come back here — every answer is recorded automatically."
    )
    st.stop()

stats = al.summary(events)

# --- headline metrics ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total questions", stats["total"])
c2.metric("Today", stats["today"])
c3.metric("Unique questions", stats["unique_questions"])
c4.metric("Avg response time", f"{stats['avg_ms']} ms")
c5.metric("Rule-based share", f"{stats['rule_pct']}%")

st.divider()

# --- engine split + daily activity side by side ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Engine usage")
    breakdown = al.engine_breakdown(events)
    if breakdown:
        df_eng = pd.DataFrame(
            [{"Engine": k, "Questions": v} for k, v in breakdown.items()]
        ).set_index("Engine")
        st.bar_chart(df_eng)

with col_right:
    st.subheader("Daily activity")
    daily = al.daily_activity(events)
    if daily:
        df_day = pd.DataFrame(daily).set_index("date")
        df_day.index.name = "Date"
        df_day.columns = ["Questions"]
        st.line_chart(df_day)

st.divider()

# --- top questions + top plans side by side ---
col_q, col_p = st.columns(2)

with col_q:
    st.subheader("Top 10 questions asked")
    top_q = al.top_questions(events, n=10)
    if top_q:
        df_q = pd.DataFrame(top_q)
        df_q.columns = ["Question", "Count", "Engine"]
        st.dataframe(df_q, use_container_width=True, hide_index=True,
                     column_config={
                         "Count": st.column_config.NumberColumn(width="small"),
                         "Engine": st.column_config.TextColumn(width="medium"),
                     })

with col_p:
    st.subheader("Top 10 plans mentioned")
    top_p = al.top_plans(events, n=10)
    if top_p:
        # Try to join with the live dataset to show plan names, not just codes.
        try:
            from chatbot.catalog import load_dataframe
            df_plans = load_dataframe()[["Plan_SOC_ID", "Plan_Name", "Plan_Rental"]]
            df_p = pd.DataFrame(top_p)
            df_p = df_p.merge(df_plans, left_on="code", right_on="Plan_SOC_ID",
                              how="left").drop(columns="Plan_SOC_ID")
            df_p.columns = ["Code", "Mentions", "Plan name", "Price"]
            df_p["Price"] = df_p["Price"].apply(
                lambda v: f"₹{v}" if pd.notna(v) else "")
        except Exception:
            df_p = pd.DataFrame(top_p)
            df_p.columns = ["Code", "Mentions"]
        st.dataframe(df_p, use_container_width=True, hide_index=True,
                     column_config={
                         "Mentions": st.column_config.NumberColumn(width="small"),
                     })
    else:
        st.caption("No plan codes mentioned in answers yet.")

st.divider()

# --- raw log + clear ---
with st.expander(f"Raw log ({len(events)} events)"):
    df_raw = pd.DataFrame(events)
    if "plans" in df_raw.columns:
        df_raw["plans"] = df_raw["plans"].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else str(v))
    st.dataframe(df_raw, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download log (JSONL)",
        data=open(al.LOG_FILE, encoding="utf-8").read() if al.LOG_FILE.exists() else "",
        file_name="vi_copilot_analytics.jsonl",
        mime="application/jsonl",
    )

st.divider()
st.subheader("Clear log")
st.caption("Deletes all recorded events permanently. Cannot be undone.")
if st.button("🗑️ Clear analytics log", type="secondary"):
    cleared = al.clear_log()
    st.success(f"Cleared {cleared} events.")
    st.rerun()
