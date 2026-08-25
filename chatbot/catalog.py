"""
catalog.py
==========

Loads the clean WIDE dataset (output/vi_plans_wide_dataset.csv) -- one row per
plan, no blank cells -- and turns it into the grounding block the chatbot
answers from.

TOKEN EFFICIENCY
----------------
The whole catalogue is sent to the model on EVERY question, so its size
directly drives API quota use. Two things keep it small without losing a single
fact:

  1. CSV instead of JSON. A list-of-objects repeats all 58 column names for all
     62 plans; CSV states the header once. (~48% smaller.)
  2. Constant columns are hoisted out. Ten columns hold the identical value for
     every plan (Operator, 4G_Eligible, the Activation_* and Tagging_* fields).
     Repeating them 62 times is pure waste, so they are stated once as
     "applies to every plan" facts instead.

Together: ~36,000 tokens -> ~13,700 tokens per request, same information.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
WIDE_CSV = BASE_DIR / "output" / "vi_plans_wide_dataset.csv"

# keep_default_na=False is essential: the wide dataset deliberately uses words
# like "No OTT Bundled" and "Not Applicable". Without this pandas would turn
# some of them back into NaN -- the exact "blank columns" trap we removed.
_READ_OPTS = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False)


def load_dataframe(path: Path = WIDE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Wide dataset not found at {path}.\n"
            f"Run  python build_wide_dataset.py  first to generate it."
        )
    return pd.read_csv(path, **_READ_OPTS)


def load_plan_records(path: Path = WIDE_CSV) -> list[dict]:
    """One dict per plan, every column, exactly as written in the CSV."""
    return load_dataframe(path).to_dict(orient="records")


def split_constant_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (constant_columns, varying_columns)."""
    constant = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    varying = [c for c in df.columns if c not in constant]
    return constant, varying


def build_grounding(df: pd.DataFrame) -> tuple[str, str]:
    """
    Build the two grounding pieces:
      - shared_facts: the columns identical across all plans, stated once.
      - plans_csv:    the remaining columns as CSV (header + one row per plan).
    """
    constant, varying = split_constant_columns(df)

    lines = [f"{col}: {df[col].iloc[0]}" for col in constant]
    shared_facts = "\n".join(lines)

    buf = io.StringIO()
    df[varying].to_csv(buf, index=False)
    plans_csv = buf.getvalue().strip()

    return shared_facts, plans_csv


def _to_int(value):
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def dataset_facts(df: pd.DataFrame) -> dict:
    """Headline numbers for the sidebar and the prompt snapshot."""
    prices = [p for p in (_to_int(v) for v in df.get("Plan_Rental", [])) if p is not None]
    types = sorted({t for t in df.get("Plan_Type", []) if t})
    return {
        "plan_count": len(df),
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "plan_types": types,
        "columns": list(df.columns),
    }


def get_catalog(path: Path = WIDE_CSV) -> tuple[list[dict], dict, dict]:
    """
    Load everything the app needs.

    Returns (records, grounding, facts) where grounding is
    {"shared_facts": str, "plans_csv": str} ready for the prompt builder.
    """
    df = load_dataframe(path)
    shared_facts, plans_csv = build_grounding(df)
    grounding = {"shared_facts": shared_facts, "plans_csv": plans_csv}
    return df.to_dict(orient="records"), grounding, dataset_facts(df)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    recs, grounding, facts = get_catalog()
    size = len(grounding["shared_facts"]) + len(grounding["plans_csv"])
    print("Loaded", facts["plan_count"], "plans,", len(facts["columns"]), "columns")
    print("Price range:", facts["min_price"], "-", facts["max_price"])
    print(f"Grounding size: {size} chars (~{size // 4} tokens)")
    print("\n--- shared facts (stated once) ---")
    print(grounding["shared_facts"])
    print("\n--- first 2 CSV lines ---")
    print("\n".join(grounding["plans_csv"].splitlines()[:2])[:400])
