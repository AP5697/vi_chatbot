"""
dataset_admin.py
================

The engine behind the Admin page: validate a candidate plan dataset, diff it
against the live one, and swap it in safely (with a timestamped backup and a
one-click rollback).

Everything here is pure pandas / filesystem work with NO Streamlit import, so it
can be unit-tested offline (see chatbot/test_dataset_admin.py). The page in
pages/1_Admin.py is a thin UI over these functions.

WHY VALIDATION IS STRICT
------------------------
The whole app -- both engines -- reads fixed column names straight off this CSV
(Plan_SOC_ID, Plan_Rental, Validity_Days, Daily_Data_Limit, Voice_Benefit,
5G_Eligible, OTT_Platform, Plan_Status, ...). A CSV that is missing a column, or
has blank cells, or a non-numeric price, would not error at upload -- it would
break a table render or a filter LATER, mid-call, in front of a customer. So we
refuse it up front and say exactly why, rather than let a bad file go live.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .catalog import WIDE_CSV, _READ_OPTS, load_dataframe

# Backups live beside the dataset so a rollback never depends on anything
# outside the repo/output folder.
BACKUP_DIR = WIDE_CSV.parent / "backups"

# The identity column every plan must have, and must have uniquely.
KEY_COLUMN = "Plan_SOC_ID"

# Columns the app does arithmetic / numeric comparisons on. A non-numeric value
# here silently breaks price and validity filters, so it is a hard error.
NUMERIC_COLUMNS = ("Plan_Rental", "Validity_Days")


def _num(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# Reading an uploaded file
# --------------------------------------------------------------------------

def read_uploaded_csv(data: bytes) -> pd.DataFrame:
    """
    Parse uploaded bytes into a DataFrame using the SAME options as the live
    loader (all strings, keep 'No OTT Bundled' etc. as literal text). Reading it
    any other way here than the app reads it in production would let a file pass
    validation and still behave differently once live.
    """
    return pd.read_csv(io.BytesIO(data), **_READ_OPTS)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return f"Valid — {self.row_count} plans, {self.column_count} columns."
        if self.ok:
            return f"Valid with {len(self.warnings)} warning(s)."
        return f"Rejected — {len(self.errors)} problem(s) must be fixed first."


def validate(new_df: pd.DataFrame,
             reference_columns: list[str] | None = None) -> ValidationReport:
    """
    Check a candidate dataset against every rule the app relies on.

    reference_columns defaults to the CURRENT live dataset's columns, so the new
    file must carry at least the same schema. Extra columns are allowed (a
    warning, not an error) -- adding a field never breaks an existing reader.
    """
    if reference_columns is None:
        reference_columns = list(load_dataframe().columns)

    errors: list[str] = []
    warnings: list[str] = []

    if new_df.empty:
        errors.append("The file has no data rows.")
        return ValidationReport(False, errors, warnings, 0, len(new_df.columns))

    # --- schema ---
    have = list(new_df.columns)
    missing = [c for c in reference_columns if c not in have]
    extra = [c for c in have if c not in reference_columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}.")
    if extra:
        warnings.append(f"Extra column(s) the app won't use: {', '.join(extra)}.")

    # --- key column: present, unique, non-blank ---
    if KEY_COLUMN not in have:
        errors.append(f"Missing the plan-code column '{KEY_COLUMN}'.")
    else:
        ids = new_df[KEY_COLUMN].astype(str).str.strip()
        blank_ids = int((ids == "").sum())
        if blank_ids:
            errors.append(f"{blank_ids} row(s) have a blank {KEY_COLUMN}.")
        dupes = sorted(ids[ids.duplicated() & (ids != "")].unique())
        if dupes:
            shown = ", ".join(dupes[:5]) + ("…" if len(dupes) > 5 else "")
            errors.append(f"Duplicate plan code(s): {shown}.")

    # --- no blank cells anywhere (the app assumes a fully-filled grid) ---
    blank_by_col = {
        col: int(new_df[col].astype(str).str.strip().eq("").sum())
        for col in have
    }
    offenders = {c: n for c, n in blank_by_col.items() if n}
    if offenders:
        total = sum(offenders.values())
        cols = ", ".join(f"{c} ({n})" for c, n in list(offenders.items())[:6])
        errors.append(f"{total} blank cell(s) found in: {cols}"
                      + ("…" if len(offenders) > 6 else "") + ".")

    # --- numeric columns really parse as numbers ---
    for col in NUMERIC_COLUMNS:
        if col in have:
            bad = int(new_df[col].apply(lambda v: _num(v) is None).sum())
            if bad:
                errors.append(f"{bad} row(s) have a non-numeric {col}.")

    return ValidationReport(
        ok=not errors, errors=errors, warnings=warnings,
        row_count=len(new_df), column_count=len(have),
    )


# --------------------------------------------------------------------------
# Diff against the live dataset
# --------------------------------------------------------------------------

@dataclass
class DiffReport:
    added: list[dict]                    # [{code, name}]
    removed: list[dict]                  # [{code, name}]
    changed: list[dict]                  # [{code, name, field, old, new}]
    unchanged_count: int

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def headline(self) -> str:
        changed_plans = len({c["code"] for c in self.changed})
        return (f"{len(self.added)} added · {len(self.removed)} removed · "
                f"{changed_plans} changed · {self.unchanged_count} unchanged")


def _name_of(df: pd.DataFrame, code: str) -> str:
    row = df[df[KEY_COLUMN].astype(str) == code]
    if row.empty or "Plan_Name" not in df.columns:
        return code
    return str(row.iloc[0]["Plan_Name"])


def diff(old_df: pd.DataFrame, new_df: pd.DataFrame,
         max_field_changes: int = 200) -> DiffReport:
    """
    Compare two datasets by plan code and report what changed.

    Field-level changes are capped (max_field_changes) so a wholesale replace of
    every plan can't produce a diff table thousands of rows long -- past the cap
    the count still reflects reality, only the per-field listing stops.
    """
    old_ids = set(old_df[KEY_COLUMN].astype(str))
    new_ids = set(new_df[KEY_COLUMN].astype(str))

    added = [{"code": c, "name": _name_of(new_df, c)}
             for c in sorted(new_ids - old_ids)]
    removed = [{"code": c, "name": _name_of(old_df, c)}
               for c in sorted(old_ids - new_ids)]

    shared_cols = [c for c in old_df.columns if c in new_df.columns and c != KEY_COLUMN]
    old_by = old_df.set_index(old_df[KEY_COLUMN].astype(str))
    new_by = new_df.set_index(new_df[KEY_COLUMN].astype(str))

    changed: list[dict] = []
    unchanged = 0
    for code in sorted(new_ids & old_ids):
        o, n = old_by.loc[code], new_by.loc[code]
        row_changed = False
        for col in shared_cols:
            ov, nv = str(o[col]), str(n[col])
            if ov != nv:
                row_changed = True
                if len(changed) < max_field_changes:
                    changed.append({"code": code, "name": str(n.get("Plan_Name", code)),
                                    "field": col, "old": ov, "new": nv})
        unchanged += not row_changed

    return DiffReport(added, removed, changed, unchanged)


# --------------------------------------------------------------------------
# Apply / backup / rollback
# --------------------------------------------------------------------------

def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def backup_current(now: datetime | None = None) -> Path | None:
    """
    Copy the live CSV into output/backups/ before it is overwritten.

    Returns the backup path, or None if there is no live file yet (first-ever
    upload). Uploading is never blocked by the absence of a prior dataset.

    Two backups can land in the same second (apply then immediately restore),
    which would collide on the timestamped name and silently overwrite the first
    -- so a numeric suffix is added when the name is already taken. Without this,
    a restore could back up over the very file it is about to restore FROM.
    """
    if not WIDE_CSV.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp(now)
    dest = BACKUP_DIR / f"{WIDE_CSV.stem}_{stamp}.csv"
    counter = 2
    while dest.exists():
        dest = BACKUP_DIR / f"{WIDE_CSV.stem}_{stamp}-{counter}.csv"
        counter += 1
    shutil.copy2(WIDE_CSV, dest)
    return dest


def apply_new_dataset(new_df: pd.DataFrame,
                      now: datetime | None = None) -> Path | None:
    """
    Back up the current dataset, then write the validated new one in its place,
    using the exact encoding the loader expects. Returns the backup path (or
    None if there was nothing to back up).

    Caller MUST have validated new_df first -- this function trusts its input and
    does not re-check. It also does not clear Streamlit's cache; the page does
    that so this stays UI-free and testable.
    """
    backup = backup_current(now)
    WIDE_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(WIDE_CSV, index=False, encoding="utf-8-sig")
    return backup


@dataclass
class Backup:
    path: Path
    when: datetime
    plan_count: int

    def label(self) -> str:
        return f"{self.when:%d %b %Y, %H:%M:%S} — {self.plan_count} plans"


def list_backups() -> list[Backup]:
    """Every saved backup, newest first, with its plan count for the picker."""
    if not BACKUP_DIR.exists():
        return []
    out: list[Backup] = []
    prefix = f"{WIDE_CSV.stem}_"
    for path in BACKUP_DIR.glob(f"{prefix}*.csv"):
        # Strip the (underscore-containing) dataset name, leaving just the
        # 'YYYYMMDD-HHMMSS' stamp, plus an optional '-N' collision suffix.
        remainder = path.stem[len(prefix):]
        parts = remainder.split("-")
        try:
            when = datetime.strptime(f"{parts[0]}-{parts[1]}", "%Y%m%d-%H%M%S")
        except (ValueError, IndexError):
            when = datetime.fromtimestamp(path.stat().st_mtime)
        try:
            count = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1
        except OSError:
            count = 0
        out.append(Backup(path, when, max(count, 0)))
    return sorted(out, key=lambda b: b.when, reverse=True)


def restore_backup(path: Path, now: datetime | None = None) -> Path | None:
    """
    Roll the live dataset back to a saved backup.

    The CURRENT dataset is itself backed up first, so a rollback is undoable too
    -- an admin who restores the wrong version can always step forward again.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Backup not found: {path}")
    safety = backup_current(now)
    shutil.copy2(path, WIDE_CSV)
    return safety
