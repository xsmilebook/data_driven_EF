from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CleanupDecision:
    keep: set[str]
    remove: set[str]


def _default_decision() -> CleanupDecision:
    """
    Decisions based on the user's "keep (green highlight)" list.

    We remove the *non-keep* files from the active raw folder by moving them to an archive folder.
    """
    keep = {
        # THU_144_LSC: keep 20231203, remove 20231217
        "THU_20231203_144_LSC_李斯晨_GameData.xlsx",
        # THU_161_ZJY: keep 20240221, remove 20231230
        "THU_20240221_161_ZJY_张佳茵_GameData.xlsx",
        # THU_435_CJT: keep 20241201, remove 20241130
        "THU_20241201_435_CJT_陈嘉桐_GameData.xlsx",
        # THU_601_ZYY: keep 20250607, remove 20250605
        "THU_20250607_601_ZYY_朱洋仪_GameData.xlsx",
        # THU_602_ZWZ: keep 20250607, remove 20250613
        "THU_20250607_602_ZWZ_张威志_GameData.xlsx",
        # THU_628_ZSY: keep 20250708, remove 20250709
        "THU_20250708_628_ZSY_张烁杨_GameData.xlsx",
        # Special: keep 683_LYB, remove 684_LC_刘元博
        "THU_20250725_683_LYB_刘元博_GameData.xlsx",
    }

    remove = {
        "THU_20231217_144_LSC_李斯晨_GameData.xlsx",
        "THU_20231230_161_ZJY_张佳茵_GameData.xlsx",
        "THU_20241130_435_CJT_陈嘉桐_GameData.xlsx",
        "THU_20250605_601_ZYY_朱洋仪_GameData.xlsx",
        "THU_20250613_602_ZWZ_张威志_GameData.xlsx",
        "THU_20250709_628_ZSY_张烁杨_GameData.xlsx",
        "THU_20250725_684_LC_刘元博_GameData.xlsx",
    }
    return CleanupDecision(keep=keep, remove=remove)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Remove duplicate APP behavioral workbooks from raw folder (keep list provided by user)."
    )
    ap.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/behavior_data/cibr_app_data",
        help="Raw workbook directory.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["move", "delete"],
        default="move",
        help="Default move to archive (safer). Use delete for permanent removal.",
    )
    ap.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Archive folder for moved files. Default: <raw_dir>/_excluded_duplicates/run_<timestamp>/",
    )
    ap.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Write action log to CSV. Default: <archive_dir>/cleanup_log.csv (when mode=move).",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform move/delete. Without this flag, only prints planned actions.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    decision = _default_decision()

    if args.archive_dir:
        archive_dir = Path(args.archive_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = raw_dir / "_excluded_duplicates" / f"run_{ts}"

    if args.log_csv:
        log_csv = Path(args.log_csv)
    else:
        log_csv = archive_dir / "cleanup_log.csv"

    planned = []
    for fname in sorted(decision.remove):
        src = raw_dir / fname
        if not src.exists():
            planned.append(("missing", str(src), "", "not_found"))
            continue
        if args.mode == "move":
            dst = archive_dir / fname
            planned.append(("move", str(src), str(dst), "duplicate_removed_keep_green"))
        else:
            planned.append(("delete", str(src), "", "duplicate_removed_keep_green"))

    print(f"raw_dir: {raw_dir}")
    print(f"mode: {args.mode}")
    if args.mode == "move":
        print(f"archive_dir: {archive_dir}")
        print(f"log_csv: {log_csv}")
    print(f"execute: {bool(args.execute)}")
    print("")
    for action, src, dst, reason in planned:
        if action == "move":
            print(f"[PLAN] MOVE  {src} -> {dst}  ({reason})")
        elif action == "delete":
            print(f"[PLAN] DELETE {src}  ({reason})")
        else:
            print(f"[PLAN] MISSING {src}  ({reason})")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to apply.")
        return 0

    if args.mode == "move":
        archive_dir.mkdir(parents=True, exist_ok=True)

    applied: list[tuple[str, str, str, str]] = []
    had_error = False
    for action, src, dst, reason in planned:
        if action == "missing":
            applied.append((action, src, dst, reason))
            continue
        try:
            if action == "move":
                dst_p = Path(dst)
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                if dst_p.exists():
                    # A previous run may have copied the file but failed to delete the source (locked file).
                    try:
                        Path(src).unlink()
                        applied.append(("delete_after_existing_archive", src, dst, reason))
                    except PermissionError as e:
                        had_error = True
                        applied.append(("error_src_locked", src, dst, f"{reason}; {e}"))
                    continue

                try:
                    shutil.move(src, dst)
                    applied.append((action, src, dst, reason))
                except PermissionError as e:
                    # Fallback: copy into archive then try deleting the source.
                    try:
                        shutil.copy2(src, dst)
                        try:
                            Path(src).unlink()
                            applied.append(("copy_then_delete", src, dst, reason))
                        except PermissionError as e2:
                            had_error = True
                            applied.append(("copied_but_src_locked", src, dst, f"{reason}; {e2}"))
                    except Exception as e2:
                        had_error = True
                        applied.append(("error_move_failed", src, dst, f"{reason}; {e}; {e2}"))
            elif action == "delete":
                Path(src).unlink()
                applied.append((action, src, dst, reason))
        except Exception as e:
            had_error = True
            applied.append(("error", src, dst, f"{reason}; {e}"))

    if args.mode == "move":
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with log_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["action", "src", "dst", "reason"])
            w.writerows(applied)
        print(f"\nWrote log: {log_csv}")

    if had_error:
        print("\nCompleted with errors. Close any program using the remaining files and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
