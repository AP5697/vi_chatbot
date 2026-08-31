"""
test_dataset_admin.py
=====================

Offline tests for the Admin dataset engine (dataset_admin.py): validation, diff,
and the backup / apply / restore lifecycle.

The lifecycle tests must NOT touch the real output/vi_plans_wide_dataset.csv, so
they redirect the module's WIDE_CSV and BACKUP_DIR at a throwaway temp directory
for the duration of the run and restore them afterwards. Everything stays
offline and deterministic.

Run:  python -m chatbot.test_dataset_admin
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import dataset_admin as A
from .catalog import load_dataframe

# A minimal schema that mirrors the real one for the columns the app relies on.
COLS = ["Plan_SOC_ID", "Plan_Name", "Plan_Rental", "Validity_Days",
        "Voice_Benefit", "5G_Eligible"]


def _row(code, name, price, days, voice="Unlimited Calls", g5="Yes"):
    return {"Plan_SOC_ID": code, "Plan_Name": name, "Plan_Rental": price,
            "Validity_Days": days, "Voice_Benefit": voice, "5G_Eligible": g5}


def _base_df():
    return pd.DataFrame([
        _row("10000001", "Plan A", "199", "28"),
        _row("10000002", "Plan B", "299", "28"),
        _row("10000003", "Plan C", "499", "56"),
    ], columns=COLS)


class Check:
    def __init__(self, label, fn):
        self.label = label
        self.fn = fn


def _validation_checks() -> list[Check]:
    def valid_passes():
        r = A.validate(_base_df(), COLS)
        return r.ok and r.row_count == 3 and not r.errors

    def empty_rejected():
        r = A.validate(_base_df().iloc[0:0], COLS)
        return not r.ok and any("no data" in e.lower() for e in r.errors)

    def missing_column_rejected():
        r = A.validate(_base_df().drop(columns=["Plan_Rental"]), COLS)
        return not r.ok and any("Plan_Rental" in e for e in r.errors)

    def extra_column_warns_not_errors():
        df = _base_df().assign(New_Col="x")
        r = A.validate(df, COLS)
        return r.ok and any("Extra" in w for w in r.warnings)

    def duplicate_id_rejected():
        df = _base_df()
        df.loc[1, "Plan_SOC_ID"] = "10000001"
        r = A.validate(df, COLS)
        return not r.ok and any("Duplicate" in e for e in r.errors)

    def blank_cell_rejected():
        df = _base_df()
        df.loc[0, "Voice_Benefit"] = "   "
        r = A.validate(df, COLS)
        return not r.ok and any("blank cell" in e.lower() for e in r.errors)

    def blank_id_rejected():
        df = _base_df()
        df.loc[0, "Plan_SOC_ID"] = ""
        r = A.validate(df, COLS)
        return not r.ok and any("blank" in e.lower() for e in r.errors)

    def nonnumeric_price_rejected():
        df = _base_df()
        df.loc[2, "Plan_Rental"] = "free"
        r = A.validate(df, COLS)
        return not r.ok and any("non-numeric Plan_Rental" in e for e in r.errors)

    def nonnumeric_validity_rejected():
        df = _base_df()
        df.loc[1, "Validity_Days"] = "N/A"
        r = A.validate(df, COLS)
        return not r.ok and any("non-numeric Validity_Days" in e for e in r.errors)

    def real_dataset_is_valid():
        live = load_dataframe()
        r = A.validate(live, list(live.columns))
        return r.ok

    return [
        Check("valid file passes", valid_passes),
        Check("empty file rejected", empty_rejected),
        Check("missing column rejected", missing_column_rejected),
        Check("extra column warns, not errors", extra_column_warns_not_errors),
        Check("duplicate plan code rejected", duplicate_id_rejected),
        Check("blank cell rejected", blank_cell_rejected),
        Check("blank plan code rejected", blank_id_rejected),
        Check("non-numeric price rejected", nonnumeric_price_rejected),
        Check("non-numeric validity rejected", nonnumeric_validity_rejected),
        Check("the live dataset passes validation", real_dataset_is_valid),
    ]


def _diff_checks() -> list[Check]:
    def detects_added():
        new = pd.concat([_base_df(),
                         pd.DataFrame([_row("10000009", "Plan Z", "999", "84")],
                                      columns=COLS)], ignore_index=True)
        d = A.diff(_base_df(), new)
        return len(d.added) == 1 and d.added[0]["code"] == "10000009"

    def detects_removed():
        d = A.diff(_base_df(), _base_df().iloc[:-1])
        return len(d.removed) == 1 and d.removed[0]["code"] == "10000003"

    def detects_field_change():
        new = _base_df()
        new.loc[0, "Plan_Rental"] = "249"
        d = A.diff(_base_df(), new)
        return (len(d.changed) == 1 and d.changed[0]["field"] == "Plan_Rental"
                and d.changed[0]["old"] == "199" and d.changed[0]["new"] == "249")

    def identical_has_no_changes():
        d = A.diff(_base_df(), _base_df())
        return not d.has_changes and d.unchanged_count == 3

    return [
        Check("diff detects an added plan", detects_added),
        Check("diff detects a removed plan", detects_removed),
        Check("diff detects a field change", detects_field_change),
        Check("identical datasets show no changes", identical_has_no_changes),
    ]


def _lifecycle_checks() -> list[Check]:
    """Backup / apply / restore, sandboxed to a temp dir (never the real CSV)."""

    def run_in_sandbox(fn):
        saved_csv, saved_dir = A.WIDE_CSV, A.BACKUP_DIR
        with tempfile.TemporaryDirectory() as tmp:
            A.WIDE_CSV = Path(tmp) / "live.csv"
            A.BACKUP_DIR = Path(tmp) / "backups"
            try:
                return fn()
            finally:
                A.WIDE_CSV, A.BACKUP_DIR = saved_csv, saved_dir

    def apply_writes_and_backs_up():
        def body():
            _base_df().to_csv(A.WIDE_CSV, index=False, encoding="utf-8-sig")
            new = _base_df()
            new.loc[0, "Plan_Rental"] = "249"
            backup = A.apply_new_dataset(new)
            live = pd.read_csv(A.WIDE_CSV, dtype=str)
            return (backup is not None and backup.exists()
                    and str(live.loc[0, "Plan_Rental"]) == "249")
        return run_in_sandbox(body)

    def first_apply_has_no_backup():
        def body():
            # No live file yet -> apply creates one, backup is None.
            backup = A.apply_new_dataset(_base_df())
            return backup is None and A.WIDE_CSV.exists()
        return run_in_sandbox(body)

    def restore_brings_back_original():
        def body():
            _base_df().to_csv(A.WIDE_CSV, index=False, encoding="utf-8-sig")
            original = A.WIDE_CSV.read_bytes()
            changed = _base_df()
            changed.loc[0, "Plan_Rental"] = "888"
            backup = A.apply_new_dataset(changed)
            A.restore_backup(backup)
            return A.WIDE_CSV.read_bytes() == original
        return run_in_sandbox(body)

    def same_second_backups_do_not_collide():
        def body():
            _base_df().to_csv(A.WIDE_CSV, index=False, encoding="utf-8-sig")
            fixed = datetime(2026, 8, 31, 10, 0, 0)
            b1 = A.backup_current(fixed)
            b2 = A.backup_current(fixed)   # same timestamp on purpose
            return b1 != b2 and b1.exists() and b2.exists()
        return run_in_sandbox(body)

    def list_backups_parses_label():
        def body():
            _base_df().to_csv(A.WIDE_CSV, index=False, encoding="utf-8-sig")
            A.backup_current(datetime(2026, 8, 31, 9, 30, 15))
            backups = A.list_backups()
            return (len(backups) == 1
                    and backups[0].when == datetime(2026, 8, 31, 9, 30, 15)
                    and backups[0].plan_count == 3)
        return run_in_sandbox(body)

    return [
        Check("apply writes new data and backs up the old", apply_writes_and_backs_up),
        Check("first-ever apply has no backup", first_apply_has_no_backup),
        Check("restore brings back the original bytes", restore_brings_back_original),
        Check("same-second backups don't collide", same_second_backups_do_not_collide),
        Check("list_backups parses timestamp + count", list_backups_parses_label),
    ]


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    checks = _validation_checks() + _diff_checks() + _lifecycle_checks()
    passed = 0
    print(f"Running {len(checks)} dataset-admin tests...\n")
    for i, check in enumerate(checks, 1):
        try:
            ok = bool(check.fn())
            reason = "" if ok else "returned False"
        except Exception as exc:
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{i:>2}/{len(checks)}] {status}  {check.label}")
        if not ok:
            print(f"        {reason}")

    print(f"\nRESULT: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(run())
