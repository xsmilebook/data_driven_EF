#!/usr/bin/env python3
"""
Find repeated Psychopy CSV logs per subject×task under task_psych.

Writes a TSV report listing subject folders that contain multiple CSVs for the
same task (nback/sst/switch).

Example:
  python temp/find_repeated_task_psych_csvs.py \
    --task-psych-dir data/raw/MRI_data/task_psych \
    --out-dir data/interim/MRI_data/repeated_task_psych_file
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TASK_KEYS = ("nback", "sst", "switch")

_PSYCHOPY_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})\.(\d{2})\.(\d{3})(?=\.|$)")


def _psychopy_timestamp_from_filename(name: str) -> datetime | None:
    m = _PSYCHOPY_TS_RE.search(name)
    if not m:
        return None
    date_s, hh, mm, ss, ms = m.groups()
    try:
        base = datetime.strptime(date_s, "%Y-%m-%d")
        return base.replace(
            hour=int(hh),
            minute=int(mm),
            second=int(ss),
            microsecond=int(ms) * 1000,
        )
    except Exception:
        return None


def _choose_newest_by_filename_timestamp(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, datetime, float]:
        ts = _psychopy_timestamp_from_filename(p.name)
        if ts is None:
            return (0, datetime.min, p.stat().st_mtime)
        return (1, ts, p.stat().st_mtime)

    return sorted(set(paths), key=key, reverse=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task-psych-dir",
        default="data/raw/MRI_data/task_psych",
        help="Path to task_psych directory (default: data/raw/MRI_data/task_psych).",
    )
    ap.add_argument(
        "--out-dir",
        default="data/interim/MRI_data/repeated_task_psych_file",
        help="Output directory (default: data/interim/MRI_data/repeated_task_psych_file).",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSVs recursively within each subject folder (default: false).",
    )
    return ap.parse_args()


def _extract_subject_label(folder_name: str) -> str:
    m = re.match(r"^sub-([A-Za-z0-9]+)$", folder_name)
    if m:
        return m.group(1)

    m = re.match(r"^THU_(\d{8})_(\d+)_([A-Za-z]+)$", folder_name)
    if m:
        date8, num, code = m.groups()
        return f"THU{date8}{num}{code.upper()}"

    m = re.match(r"^XY_(\d{8})_(\d+)_([A-Za-z0-9_]+)$", folder_name)
    if m:
        date8, num, code = m.groups()
        return f"XY{date8}{num}{code.replace('_', '').upper()}"

    return folder_name


def _detect_task_from_filename(name: str) -> str | None:
    lower = name.lower()
    if "_nback_" in lower:
        return "nback"
    if "_switch_" in lower:
        return "switch"
    if "_sst_" in lower:
        return "sst"
    return None


@dataclass(frozen=True)
class RepeatedRecord:
    subject_folder: str
    subject_label: str
    task: str
    n_csv: int
    newest_csv: str
    csv_files: list[str]


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    task_psych_dir = (repo_root / args.task_psych_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not task_psych_dir.exists():
        raise FileNotFoundError(f"task_psych_dir not found: {task_psych_dir}")

    records: list[RepeatedRecord] = []
    subjects_seen = 0

    for subject_dir in sorted(p for p in task_psych_dir.iterdir() if p.is_dir()):
        subjects_seen += 1
        subject_folder = subject_dir.name
        subject_label = _extract_subject_label(subject_folder)

        csvs = list(subject_dir.rglob("*.csv") if args.recursive else subject_dir.glob("*.csv"))
        by_task: dict[str, list[Path]] = {k: [] for k in TASK_KEYS}
        for p in csvs:
            task = _detect_task_from_filename(p.name)
            if task is not None:
                by_task[task].append(p)

        for task, files in by_task.items():
            if len(files) <= 1:
                continue
            files_sorted = _choose_newest_by_filename_timestamp(files)
            rels = [str(p.relative_to(task_psych_dir)).replace("\\", "/") for p in files_sorted]
            records.append(
                RepeatedRecord(
                    subject_folder=subject_folder,
                    subject_label=subject_label,
                    task=task,
                    n_csv=len(files_sorted),
                    newest_csv=rels[0],
                    csv_files=rels,
                )
            )

    tsv_path = out_dir / "repeated_task_psych_files.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "\t".join(
                [
                    "subject_folder",
                    "subject_label",
                    "task",
                    "n_csv",
                    "newest_csv",
                    "csv_files",
                ]
            )
            + "\n"
        )
        for r in records:
            f.write(
                "\t".join(
                    [
                        r.subject_folder,
                        r.subject_label,
                        r.task,
                        str(r.n_csv),
                        r.newest_csv,
                        ";".join(r.csv_files),
                    ]
                )
                + "\n"
            )

    meta_path = out_dir / "README.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"created_at={datetime.now().isoformat(timespec='seconds')}",
                f"task_psych_dir={task_psych_dir}",
                f"subjects_scanned={subjects_seen}",
                f"repeated_subject_task_pairs={len(records)}",
                f"report={tsv_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"task_psych_dir={task_psych_dir}")
    print(f"out_dir={out_dir}")
    print(f"subjects_scanned={subjects_seen}")
    print(f"repeated_subject_task_pairs={len(records)}")
    print(f"report={tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

