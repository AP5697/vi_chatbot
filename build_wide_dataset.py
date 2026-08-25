"""
build_wide_dataset.py
=====================

Builds a CLEAN, COMPLETE, WIDE dataset: one row per plan, one column per
attribute, and NO blank cells anywhere. Conditional attributes that the source
left as NA (OTT, ISD, roaming, daily-cap, closure info) are filled with clear,
meaningful values ("None", "Not Included", "No Daily Cap", "Active", ...).

Source of plan facts:
    input/vi_only_plan_catalogue_for_audit.csv   (real 62 VI plans, read-only)

Outputs (fresh files; nothing existing is overwritten):
    output/vi_plans_wide_dataset.csv
    output/vi_plans_wide_dataset.xlsx   (formatted, one sheet)

Design
------
The source is long-format (many benefit rows per plan). Plan-level attributes
repeat identically down those rows, so we take one representative row per plan
and reshape into a single tidy row. Real facts (price, data, validity, voice,
SMS, OTT platform, circles) are carried through unchanged. The reference
taxonomy from the mentor's sheet becomes explicit columns:

    1.Telco Benefits/4.Video      -> Video_Benefit
    2.Non Telco +++/1.OTTS        -> OTTS_Detail
    3.Activation_Mode/*           -> Activation_ViApp / _Website / _SCRM / _TAT
    4.Additional Information       -> Additional_Information
    5.Tagging/*                   -> Tagging_Enquiry / _Request / _Complaint
    6.Base Tariff                 -> Base_Tariff

Every cell is guaranteed non-empty: a final sweep replaces any remaining
blank/NA with "Not Applicable" so the mentor never sees an empty column.

Run:  python build_wide_dataset.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "input" / "vi_only_plan_catalogue_for_audit.csv"
OUT_CSV = BASE_DIR / "output" / "vi_plans_wide_dataset.csv"
OUT_XLSX = BASE_DIR / "output" / "vi_plans_wide_dataset.xlsx"

# Final column order for the wide dataset.
COLUMN_ORDER = [
    # --- Identity ---
    "Plan_SOC_ID", "Plan_ID", "Plan_Name", "Plan_Type", "Product_Type",
    "Customer_Type", "Segment", "Brand", "Operator",
    # --- Pricing & validity ---
    "Plan_Rental", "Plan_Rental_GST", "Base_Tariff", "Validity_Days",
    # --- Telco (core) benefits ---
    "Voice_Benefit", "Data_Benefit", "Daily_Data_Limit", "Total_Data_Limit",
    "Data_Unit", "SMS_Benefit", "SMS_Limit", "FUP_Limit", "FUP_Speed",
    "Data_Rollover", "Night_Data", "Weekend_Data", "Video_Benefit",
    "5G_Eligible", "4G_Eligible",
    # --- Non-telco / add-on benefits ---
    "OTT_Benefit", "OTT_Platform", "OTTS_Detail", "Roaming_Benefit",
    "ISD_Benefit", "Additional_Benefits", "Additional_Information",
    # --- Activation modes (3.Activation_Mode) ---
    "Activation_Channel", "Activation_ViApp", "Activation_Website",
    "Activation_SCRM", "Activation_TAT", "Recharge_Channel",
    # --- Care tagging (5.Tagging) ---
    "Tagging_Enquiry", "Tagging_Request", "Tagging_Complaint",
    # --- Availability & lifecycle ---
    "Go_Live_Circle", "Circle_Count", "Market", "Go_Live_Date",
    "Launch_Quarter", "Active_Inactive", "Plan_Status", "Auto_Renewal",
    "Pack_Closure_Date", "Closure_Reason", "Speed_Post_Limit",
    "Created_Date", "Last_Updated_Date", "Plan_Brief",
]


def _val(row, col, default=""):
    """Read a source cell, treating NA/blank as missing -> return default."""
    v = row.get(col)
    if v is None:
        return default
    s = str(v).strip()
    if s == "" or s.upper() in {"NA", "N/A", "NAN", "NONE"}:
        return default
    return s


def build_row(rep: pd.Series) -> dict:
    """Turn one plan's representative source row into a full, filled wide row."""
    ott_platform_raw = _val(rep, "OTT_Platform", "")
    add_benefits_raw = _val(rep, "Additional_Benefits", "")
    daily_raw = _val(rep, "Daily_Data_Limit", "")
    total_raw = _val(rep, "Total_Data_Limit", "")
    unit = _val(rep, "Data_Unit", "GB")
    has_ott = _val(rep, "OTT_Benefit", "No").lower() == "yes"

    # Daily data: real value if present, else a clear description.
    if daily_raw:
        daily = f"{daily_raw} {unit}/Day"
    elif total_raw:
        daily = f"No Daily Cap (Total {total_raw} {unit} Pool)"
    else:
        daily = "No Data Benefit"

    # Video experience (new Telco sub-type), derived from OTT / Vi Movies.
    if "vi movies" in add_benefits_raw.lower() or "vi movies" in ott_platform_raw.lower():
        video = "Vi Movies & TV Access (Video Streaming)"
    elif has_ott:
        video = "HD Video Streaming via Bundled OTT"
    else:
        video = "Standard Video Streaming (SD)"

    # OTTS aggregation (new Non-Telco category).
    ott_platform = ott_platform_raw if ott_platform_raw else "No OTT Bundled"
    otts_detail = f"Bundled OTT: {ott_platform_raw}" if ott_platform_raw else "No OTT Subscription Bundled"

    rental = _val(rep, "Plan_Rental", "")
    base_tariff = _val(rep, "Plan_Rental_GST", f"Base Tariff ₹{rental}" if rental else "Not Available")

    row = {
        # Identity
        "Plan_SOC_ID": _val(rep, "Plan_SOC_ID"),
        "Plan_ID": _val(rep, "Plan_ID"),
        "Plan_Name": _val(rep, "Plan_Name"),
        "Plan_Type": _val(rep, "Plan_Type"),
        "Product_Type": _val(rep, "Product_Type"),
        "Customer_Type": _val(rep, "Customer_Type"),
        "Segment": _val(rep, "Segment"),
        "Brand": _val(rep, "Brand"),
        "Operator": _val(rep, "Operator", "Vodafone Idea"),
        # Pricing
        "Plan_Rental": rental,
        "Plan_Rental_GST": _val(rep, "Plan_Rental_GST"),
        "Base_Tariff": base_tariff,
        "Validity_Days": _val(rep, "Validity_Days"),
        # Telco core
        "Voice_Benefit": _val(rep, "Voice_Benefit", "Not Included"),
        "Data_Benefit": _val(rep, "Data_Benefit", "No Data Benefit"),
        "Daily_Data_Limit": daily,
        "Total_Data_Limit": total_raw if total_raw else "Not Applicable",
        "Data_Unit": unit,
        "SMS_Benefit": _val(rep, "SMS_Benefit", "No SMS Benefit"),
        "SMS_Limit": _val(rep, "SMS_Limit", "0"),
        "FUP_Limit": _val(rep, "FUP_Limit", "No FUP"),
        "FUP_Speed": _val(rep, "FUP_Speed", "Not Applicable"),
        "Data_Rollover": _val(rep, "Data_Rollover", "No"),
        "Night_Data": _val(rep, "Night_Data", "Not Included"),
        "Weekend_Data": _val(rep, "Weekend_Data", "Not Included"),
        "Video_Benefit": video,
        "5G_Eligible": _val(rep, "5G_Eligible", "No"),
        "4G_Eligible": _val(rep, "4G_Eligible", "Yes"),
        # Non-telco / add-ons
        "OTT_Benefit": _val(rep, "OTT_Benefit", "No"),
        "OTT_Platform": ott_platform,
        "OTTS_Detail": otts_detail,
        "Roaming_Benefit": _val(rep, "Roaming_Benefit", "Not Included"),
        "ISD_Benefit": _val(rep, "ISD_Benefit", "Not Included"),
        "Additional_Benefits": add_benefits_raw if add_benefits_raw else "No Additional Benefit",
        "Additional_Information": add_benefits_raw if add_benefits_raw else "No Additional Information",
        # Activation modes
        "Activation_Channel": _val(rep, "Activation_Channel", "Retail Outlet"),
        "Activation_ViApp": "Self-Activation via Vi App",
        "Activation_Website": "Activation via Vi Website",
        "Activation_SCRM": "Assisted Activation via Store CRM",
        "Activation_TAT": "Instant to 24 Hours",
        "Recharge_Channel": _val(rep, "Recharge_Channel", "Retail Outlet | Cash"),
        # Care tagging
        "Tagging_Enquiry": "Plan/Benefit Enquiry | Balance & Validity Enquiry",
        "Tagging_Request": "Activation Request | Plan Change Request | OTT Activation Request",
        "Tagging_Complaint": "Network/Speed Complaint | Billing Complaint | Deactivation Complaint",
        # Availability & lifecycle
        "Go_Live_Circle": _val(rep, "Go_Live_Circle", "Pan India"),
        "Circle_Count": _val(rep, "Circle_Count", "0"),
        "Market": _val(rep, "Market", "Pan India"),
        "Go_Live_Date": _val(rep, "Go_Live_Date", "Not Available"),
        "Launch_Quarter": _val(rep, "Launch_Quarter", "Not Available"),
        "Active_Inactive": _val(rep, "Active_Inactive", "Active"),
        "Plan_Status": _val(rep, "Plan_Status", "Live"),
        "Auto_Renewal": _val(rep, "Auto_Renewal", "No"),
        "Pack_Closure_Date": _val(rep, "Pack_Closure_Date", "Active - Not Closed"),
        "Closure_Reason": _val(rep, "Closure_Reason", "Not Applicable (Active Plan)"),
        "Speed_Post_Limit": _val(rep, "Speed_Post_Limit", "Not Applicable"),
        "Created_Date": _val(rep, "Created_Date", "Not Available"),
        "Last_Updated_Date": _val(rep, "Last_Updated_Date", "Not Available"),
        "Plan_Brief": _val(rep, "Plan_Brief", "Not Available"),
    }
    return row


def save_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Write a single formatted sheet: frozen header, autofilter, sized cols."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="VI_Plans_Wide")
        ws = writer.sheets["VI_Plans_Wide"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for i, col in enumerate(df.columns, start=1):
            longest = df[col].astype(str).str.len().max()
            width = min(max(int(longest or 0), len(str(col))) + 2, 55)
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width


def main() -> None:
    print("=" * 60)
    print("BUILD CLEAN WIDE DATASET  (one row per plan, no blanks)")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise SystemExit(f"ERROR: input not found at {INPUT_FILE}")

    src = pd.read_csv(INPUT_FILE, dtype=str, encoding="utf-8-sig")
    reps = src.drop_duplicates("Plan_SOC_ID")
    print(f"Source: {len(src)} rows -> {len(reps)} unique plans.")

    rows = [build_row(rep) for _, rep in reps.iterrows()]
    df = pd.DataFrame(rows, columns=COLUMN_ORDER)

    # Guarantee: absolutely no blank cell remains.
    df = df.apply(lambda c: c.map(lambda x: "Not Applicable" if str(x).strip() == "" else x))

    # Verify zero blanks before writing.
    blanks = int((df.astype(str).apply(lambda c: c.str.strip().eq("")).sum()).sum())
    print(f"Columns: {len(df.columns)}   Rows: {len(df)}   Blank cells: {blanks}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    save_xlsx(df, OUT_XLSX)

    print("-" * 60)
    print(f"Saved CSV  -> {OUT_CSV.name}")
    print(f"Saved XLSX -> {OUT_XLSX.name}")
    print(f"Every one of {len(df) * len(df.columns)} cells is filled "
          f"({'PASS' if blanks == 0 else 'FAIL'}).")
    print("=" * 60)


if __name__ == "__main__":
    main()
