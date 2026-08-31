"""
pages/1_Admin.py  --  Dataset admin (Streamlit multipage)
=========================================================

A non-technical operator can replace the live VI plan dataset here without
touching code or redeploying:

    upload CSV -> validate -> review the diff -> confirm -> live (auto-backup)

plus a one-click rollback to any earlier version. All the real work lives in
chatbot/dataset_admin.py (pure, unit-tested); this file is only the UI and the
passcode gate.

Streamlit auto-lists any file under pages/ in the sidebar, so this appears as a
second page beneath the main chatbot automatically.
"""

from __future__ import annotations

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from chatbot import dataset_admin as admin
from chatbot.catalog import load_dataframe
from chatbot.llm import _get_setting

st.set_page_config(page_title="VI Copilot · Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Dataset Admin")
st.caption("Replace the live VI plan dataset safely — validated, diffed, "
           "backed up, and reversible.")


# --------------------------------------------------------------------------
# Passcode gate
#
# When ADMIN_PASSCODE is configured (env var or Streamlit secret) the page is
# locked until it is entered. When it is NOT set, the page stays usable — so a
# local demo works out of the box — but shows a standing warning to set one
# before the app is shared publicly.
# --------------------------------------------------------------------------

def gate() -> bool:
    passcode = (_get_setting("ADMIN_PASSCODE") or "").strip()
    if not passcode:
        st.warning(
            "**No admin passcode set.** This page can overwrite the live "
            "dataset. Before sharing the app publicly, set `ADMIN_PASSCODE` "
            "(local: in `.env`; Streamlit Cloud: Settings → Secrets) to lock it."
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


def refresh_app_cache():
    """Drop the cached dataset/engines so both pages reload the new file."""
    st.cache_resource.clear()
    st.cache_data.clear()


# --------------------------------------------------------------------------
# Upload + validate + diff + confirm
# --------------------------------------------------------------------------

def render_upload():
    st.subheader("Upload a new dataset")

    try:
        live = load_dataframe()
        st.caption(f"Currently live: **{len(live)}** plans, "
                   f"**{len(live.columns)}** columns.")
    except FileNotFoundError:
        live = None
        st.caption("No dataset is live yet — this upload will create the first one.")

    uploaded = st.file_uploader(
        "Plan dataset (CSV, same columns as the current one)", type=["csv"],
        help="Generate this with build_wide_dataset.py, or export/edit the "
             "existing CSV keeping every column.",
    )
    if uploaded is None:
        st.session_state.pop("staged", None)
        return

    try:
        new_df = admin.read_uploaded_csv(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read that file as CSV: `{exc}`")
        return

    reference = list(live.columns) if live is not None else None
    report = admin.validate(new_df, reference)

    # --- validation result ---
    if report.ok:
        st.success(report.summary())
    else:
        st.error(report.summary())
        for e in report.errors:
            st.markdown(f"- ❌ {e}")
    for w in report.warnings:
        st.markdown(f"- ⚠️ {w}")

    if not report.ok:
        st.session_state.pop("staged", None)
        st.info("Fix the problems above and re-upload. The live dataset is untouched.")
        return

    # --- diff ---
    st.divider()
    st.subheader("What will change")
    if live is None:
        st.info(f"First dataset — all {len(new_df)} plans will be added.")
    else:
        d = admin.diff(live, new_df)
        st.markdown(f"**{d.headline()}**")
        if not d.has_changes:
            st.info("This file is identical to the live dataset — nothing to apply.")
            st.session_state.pop("staged", None)
            return
        _render_diff(d)

    # --- confirm ---
    st.divider()
    st.session_state.staged = new_df
    st.warning("Applying will replace the live dataset for **all users**. "
               "The current version is backed up automatically first.")
    if st.button("✅ Apply this dataset", type="primary"):
        backup = admin.apply_new_dataset(new_df)
        refresh_app_cache()
        st.session_state.pop("staged", None)
        st.success(f"Live dataset updated to {len(new_df)} plans."
                   + (f" Previous version saved to `{backup.name}`." if backup else ""))
        st.balloons()


def _render_diff(d: admin.DiffReport):
    c1, c2, c3 = st.columns(3)
    c1.metric("Added", len(d.added))
    c2.metric("Removed", len(d.removed))
    c3.metric("Changed", len({c["code"] for c in d.changed}))

    if d.added:
        with st.expander(f"➕ {len(d.added)} added"):
            st.dataframe(d.added, use_container_width=True, hide_index=True)
    if d.removed:
        with st.expander(f"➖ {len(d.removed)} removed"):
            st.dataframe(d.removed, use_container_width=True, hide_index=True)
    if d.changed:
        plans = len({c["code"] for c in d.changed})
        with st.expander(f"✏️ {plans} plan(s) with field changes"):
            st.dataframe(d.changed, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Rollback
# --------------------------------------------------------------------------

def render_rollback():
    st.subheader("Roll back to a previous version")
    backups = admin.list_backups()
    if not backups:
        st.caption("No backups yet. One is saved automatically each time you "
                   "apply a new dataset.")
        return

    labels = {b.label(): b for b in backups}
    choice = st.selectbox("Saved versions (newest first)", list(labels))
    picked = labels[choice]
    st.caption(f"File: `{picked.path.name}`")

    if st.button("↩️ Restore this version"):
        safety = admin.restore_backup(picked.path)
        refresh_app_cache()
        st.success(f"Restored **{picked.plan_count}** plans from {picked.label()}."
                   + (f" The version you replaced was saved as `{safety.name}`, "
                      "so this is undoable too." if safety else ""))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if gate():
    tab_upload, tab_rollback = st.tabs(["⬆️ Update dataset", "↩️ Rollback"])
    with tab_upload:
        render_upload()
    with tab_rollback:
        render_rollback()
