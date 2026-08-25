"""
segregate_telecom_data.py
=========================

Splits a telecom master plan catalogue into three separate Excel datasets.

The master file is opened READ-ONLY and is never modified.

Run:      python segregate_telecom_data.py
Requires: pip install pandas openpyxl
"""

from pathlib import Path
import os
import sys

import pandas as pd


# =====================================================================
# 1. CONFIGURATION
#    Everything you might want to change lives here, so you never have
#    to hunt through the code below to adjust a requirement.
# =====================================================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# The master file. Either .csv or .xlsx works.
# Set this to an exact filename if you want to pin it, e.g.:
#   MASTER_FILE = INPUT_DIR / "vi_only_plan_catalogue_for_audit.csv"
# Leave as None to auto-detect the (single) file sitting in input/.
MASTER_FILE = None
SHEET_NAME = 0  # only used when the master is an Excel file


def resolve_master_file(configured: Path | None) -> Path:
    """
    Work out which file to load.

    If MASTER_FILE was set explicitly above, use it (and fail loudly if
    it's missing). Otherwise, look inside input/ and pick the single
    .csv/.xlsx file found there - this avoids the whole class of "the
    filename in the script doesn't match the filename on disk" errors.
    """
    if configured is not None:
        if not configured.exists():
            sys.exit(
                f"ERROR: master file not found at {configured}\n"
                f"Files actually in {INPUT_DIR}: "
                f"{[f.name for f in INPUT_DIR.glob('*')] if INPUT_DIR.exists() else '(input/ folder does not exist)'}"
            )
        return configured

    if not INPUT_DIR.exists():
        sys.exit(f"ERROR: input folder not found at {INPUT_DIR}")

    candidates = [f for f in INPUT_DIR.glob("*") if f.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls")]

    if not candidates:
        sys.exit(f"ERROR: no .csv or .xlsx file found in {INPUT_DIR}")
    if len(candidates) > 1:
        names = ", ".join(f.name for f in candidates)
        sys.exit(
            f"ERROR: multiple data files found in {INPUT_DIR} ({names}).\n"
            f"Set MASTER_FILE explicitly at the top of the script to pick one."
        )

    found = candidates[0]
    print(f"Auto-detected master file: {found.name}")
    return found

# ---------------------------------------------------------------------
# COLUMN SELECTION - using the ORIGINAL master column names, unchanged.
# Each list is simply which columns from the master go into that dataset.
# ---------------------------------------------------------------------

DATASET_1_COLUMNS = [
    "Plan_SOC_ID",
    "Plan_Name",
    "Voice_Benefit",
    "Total_Data_Limit",
    "Daily_Data_Limit",
    "Data_Benefit",
    "SMS_Limit",
    "SMS_Benefit",
    # Added for the sales-chatbot use case - price, validity, and
    # classification fields a customer-care agent needs to actually
    # quote and compare plans. All confirmed plan-level (1 value/plan).
    "Plan_Rental",
    "Plan_Rental_GST",
    "Validity_Days",
    "Plan_Type",
    "Product_Type",
    "Customer_Type",
    "Segment",
    "Active_Inactive",
    "5G_Eligible",
    "4G_Eligible",
]

DATASET_2_COLUMNS = [
    "Plan_SOC_ID",
    "Go_Live_Circle",   # a comma-separated LIST -> gets exploded, one circle per row
]

DATASET_3_COLUMNS = [
    "Plan_SOC_ID",
    "Category_Value",
    "Category",
    "OTT_Platform",
    "Plan_Brief",
    # NOT INCLUDED (absent from this master file):
    #   Selection Type (Included/Chosen) - no such column exists
    #   Worth (benefit monetary value)   - no such column exists
]

# Columns that must stay TEXT so Excel cannot eat leading zeros
# or reformat identifiers as numbers.
STRING_COLUMNS = ["Plan_SOC_ID", "Plan_ID", "Go_Live_Circle", "Circle_Count"]

# The character that separates circle codes inside Go_Live_Circle.
CIRCLE_SEPARATOR = ","

# The separator joining multiple OTT platforms inside a single
# OTT_Platform cell, e.g. "Disney+ Hotstar Mobile + SonyLIV Mobile".
OTT_SEPARATOR = " + "

# Dataset 3 filter. The spec asked for Group Options in {CYB, TRAVEL},
# but neither value exists in this file. These are the categories that
# actually represent non-core / add-on benefits.
# Set to None to keep every benefit row instead.
DATASET_3_GROUP_FILTER = [
    "2.OTT Benefits",
    "3.Roaming Benefits",
    "4.ISD Benefits",
    "5.Additional Benefits",
]

OUTPUT_WORKBOOK = OUTPUT_DIR / "telecom_datasets.xlsx"

SHEET_NAMES = {
    1: "Plan_Benefits",
    2: "Plan_Circle",
    3: "Plan_Benefits_OTT",
}


# =====================================================================
# 2. LOADING
# =====================================================================

def load_master(path: Path, sheet=0) -> pd.DataFrame:
    """
    Read the master file into a DataFrame.

    dtype=str forces EVERY column to be read as text. That is deliberate:
    it stops pandas turning "26100564" into a number or dropping a leading
    zero before we have even looked at the data. We convert the genuinely
    numeric columns back later, on purpose, one at a time.

    encoding="utf-8-sig" strips the invisible BOM character that Excel
    writes at the start of CSV files. Without it the first column name
    comes out as "\ufeffPlan_Rental" and every lookup on it fails.
    """
    if not path.exists():
        sys.exit(f"ERROR: master file not found at {path}")

    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    return df


def inspect_dataset(df: pd.DataFrame, name: str = "MASTER") -> None:
    """Print a quick structural summary so you can eyeball what loaded."""
    print(f"\n--- {name} ---")
    print(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")


def validate_required_columns(df: pd.DataFrame, columns: list, label: str) -> list:
    """
    Check that every column we want actually exists in the master.

    Returns the list with any missing entries removed, and prints a
    warning for each one. This is why the script never silently produces
    a half-empty file: you always see what could not be found.
    """
    usable = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]

    if missing:
        print(f"\n  WARNING [{label}] these columns were not found and will be skipped:")
        for m in missing:
            print(f"     - {m}")

    return usable


# =====================================================================
# 3. DATASET BUILDERS
# =====================================================================

def create_dataset_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATASET 1 - Plan Benefits (one row per PLAN).

    The master is in "long" format: each plan appears on several rows,
    one per benefit line. Every column we need here is a plan-level
    attribute that repeats identically down those rows, so we select
    the columns and then drop the exact duplicates. 508 rows collapse
    to 62 - one per plan.

    Column names are kept EXACTLY as they appear in the master file.
    """
    cols = validate_required_columns(df, DATASET_1_COLUMNS, "Dataset 1")

    out = df[cols].drop_duplicates().reset_index(drop=True)
    print(f"\n  Dataset 1: collapsed {len(df)} benefit rows -> {len(out)} unique plans")

    # Convert the genuinely numeric fields back to numbers.
    # errors="coerce" turns anything unparseable into NaN rather than
    # crashing, so a stray "N/A" cannot kill the whole run.
    for col in ["Total_Data_Limit", "Daily_Data_Limit", "SMS_Limit", "Plan_Rental", "Validity_Days"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def create_dataset_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATASET 2 - Plan to Circle relationships.

    Go_Live_Circle holds a comma-separated LIST of circles, e.g.
    "ANE, APR, BIH, DEL". One row per plan-circle pair is far more
    useful than one row containing a list, so we:
      1. reduce to unique plan + circle-list pairs,
      2. split the string into a Python list,
      3. explode() - which turns each list item into its own row.

    A plan covering 23 circles becomes 23 rows. Column names are kept
    exactly as they appear in the master file.
    """
    cols = validate_required_columns(df, DATASET_2_COLUMNS, "Dataset 2")

    out = df[cols].drop_duplicates().copy()

    out["Go_Live_Circle"] = out["Go_Live_Circle"].str.split(CIRCLE_SEPARATOR)
    out = out.explode("Go_Live_Circle")
    out["Go_Live_Circle"] = out["Go_Live_Circle"].str.strip()

    # Drop blanks created by a trailing separator, if any.
    out = out[out["Go_Live_Circle"].notna() & (out["Go_Live_Circle"] != "")]

    return out.reset_index(drop=True)


def create_dataset_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATASET 3 - Additional / OTT benefits (one row per BENEFIT line).

    Reduced version: no Selection Type or Worth columns exist in this
    master file, so they are simply absent rather than invented.

    The filter keeps only the add-on benefit categories, excluding the
    core telco lines (voice / data / SMS / validity / FUP) which are
    already covered by Dataset 1. Column names are kept exactly as they
    appear in the master file.
    """
    cols = validate_required_columns(df, DATASET_3_COLUMNS, "Dataset 3")

    out = df[cols].copy()

    if DATASET_3_GROUP_FILTER is not None and "Category" in out.columns:
        before = len(out)
        out = out[out["Category"].isin(DATASET_3_GROUP_FILTER)]
        print(f"\n  Dataset 3: group filter kept {len(out)} of {before} rows")

    # OTT_Platform is a PLAN-level field in the source data - it repeats
    # identically on every benefit row for a plan, including roaming/ISD/
    # additional-benefit rows that have nothing to do with OTT. Exploding
    # it everywhere would pointlessly duplicate those unrelated rows. So:
    # blank it out except on the actual OTT-benefit rows (Category ==
    # "2.OTT Benefits"), THEN split/explode only those - each bundled OTT
    # row (e.g. "Disney+ Hotstar Mobile + SonyLIV Mobile") becomes one row
    # per platform; every other row is left untouched, one row each.
    if "OTT_Platform" in out.columns and "Category" in out.columns:
        before = len(out)
        is_ott_row = out["Category"] == "2.OTT Benefits"
        out.loc[~is_ott_row, "OTT_Platform"] = pd.NA

        out["OTT_Platform"] = out.apply(
            lambda r: [p.strip() for p in r["OTT_Platform"].split(OTT_SEPARATOR)]
            if is_ott_row[r.name] and pd.notna(r["OTT_Platform"]) else [r["OTT_Platform"]],
            axis=1,
        )
        out = out.explode("OTT_Platform").reset_index(drop=True)
        print(f"  Dataset 3: OTT platform split (OTT rows only) -> {before} rows became {len(out)} rows")

    return out.reset_index(drop=True)


# =====================================================================
# 4. VALIDATION
# =====================================================================

def validate_output(dataset: pd.DataFrame, name: str) -> None:
    """Print missing-value counts and duplicate counts. Nothing is deleted."""
    print(f"\n{'-' * 55}\nVALIDATION: {name}\n{'-' * 55}")
    print(f"Rows: {len(dataset):,}   Columns: {len(dataset.columns)}")

    print("\nMissing values per column:")
    for col, n in dataset.isnull().sum().items():
        flag = "  <-- check" if n else ""
        print(f"  {col:<22} {n:>6}{flag}")

    dupes = dataset.duplicated().sum()
    print(f"\nFully duplicated rows: {dupes}")


def report_plan_circle_stats(dataset: pd.DataFrame) -> None:
    """The specific relationship checks the spec asked for on Dataset 2."""
    print("\nPlan-Circle relationship:")
    print(f"  Total rows:                 {len(dataset):,}")
    print(f"  Unique Plan Codes:          {dataset['Plan_SOC_ID'].nunique()}")
    print(f"  Unique Circle Codes:        {dataset['Go_Live_Circle'].nunique()}")
    print(f"  Unique Plan-Circle pairs:   {len(dataset.drop_duplicates()):,}")
    print(f"  Duplicate Plan-Circle pairs:{dataset.duplicated().sum():>6}")

    per_plan = dataset.groupby("Plan_SOC_ID").size()
    print(f"  Circles per plan: min {per_plan.min()}, max {per_plan.max()}")


def report_unique_values(dataset: pd.DataFrame, columns: list) -> None:
    """Show what actually lives in the categorical columns before export."""
    for col in columns:
        if col not in dataset.columns:
            continue
        print(f"\nUnique values in '{col}' ({dataset[col].nunique()}):")
        for val, n in dataset[col].value_counts(dropna=False).head(15).items():
            print(f"  {str(val)[:55]:<57} {n:>5}")


# =====================================================================
# 5. EXPORT
# =====================================================================

def save_workbook(datasets: dict, path: Path) -> None:
    """
    Write all three datasets into ONE .xlsx file as separate sheets,
    instead of three separate files.

    'datasets' is {sheet_name: DataFrame}. Each sheet gets: no pandas
    index column, a frozen header row, autofilter turned on, and
    columns widened to fit their content - same formatting as before,
    just all inside a single workbook now.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, dataset in datasets.items():
            dataset.to_excel(writer, index=False, sheet_name=sheet_name)

            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # Widen each column to roughly its longest value, capped at 55.
            for i, col in enumerate(dataset.columns, start=1):
                longest = dataset[col].astype(str).str.len().max()
                width = min(max(int(longest or 0), len(str(col))) + 2, 55)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

    print(f"  Saved -> {path.name}  (sheets: {', '.join(datasets.keys())})")


def open_output_folder(path: Path) -> None:
    """Open the output folder in File Explorer (Windows) so the file
    is one click away instead of requiring a manual navigate-to."""
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
    except Exception:
        pass  # never let a convenience feature crash the real work


# =====================================================================
# 6. MAIN
# =====================================================================

def main() -> None:
    print("=" * 55)
    print("TELECOM DATA SEGREGATION")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_path = resolve_master_file(MASTER_FILE)
    master = load_master(master_path, SHEET_NAME)
    inspect_dataset(master)

    ds1 = create_dataset_1(master)
    ds2 = create_dataset_2(master)
    ds3 = create_dataset_3(master)

    validate_output(ds1, "DATASET 1 - PLAN BENEFITS")
    validate_output(ds2, "DATASET 2 - PLAN CIRCLE")
    report_plan_circle_stats(ds2)
    validate_output(ds3, "DATASET 3 - PLAN BENEFITS / OTT")
    report_unique_values(ds3, ["Category", "OTT_Platform"])

    print(f"\n{'-' * 55}\nEXPORTING\n{'-' * 55}")
    save_workbook(
        {
            SHEET_NAMES[1]: ds1,
            SHEET_NAMES[2]: ds2,
            SHEET_NAMES[3]: ds3,
        },
        OUTPUT_WORKBOOK,
    )

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Input:  {master_path.name}  ({len(master):,} rows x {len(master.columns)} cols)")
    print(f"Output: {OUTPUT_WORKBOOK.name}")
    print(f"  Sheet '{SHEET_NAMES[1]}': {len(ds1):,} rows x {len(ds1.columns)} cols")
    print(f"  Sheet '{SHEET_NAMES[2]}': {len(ds2):,} rows x {len(ds2.columns)} cols")
    print(f"  Sheet '{SHEET_NAMES[3]}': {len(ds3):,} rows x {len(ds3.columns)} cols")
    print("=" * 55)
    print(f"\nFile is ready at:\n  {OUTPUT_WORKBOOK.resolve()}")
    print("=" * 55)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 55)

    open_output_folder(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()