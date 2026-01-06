#!/usr/bin/env python3
"""
Build a subject list for task-fMRI based on Psychopy behavior logs under task_psych_dir.

Default output:
  data/processed/table/sublist/taskfmri_sublist.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Output path for taskfmri_sublist.txt")
    ap.add_argument("--require-all", action="store_true", help="Require nback+sst+switch logs per subject.")
    ap.add_argument("--dataset", type=str, default="EFNY_THU")
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument(
        "--dataset-config",
        dest="dataset_config",
        type=str,
        default="configs/dataset_tsinghua_taskfmri.yaml",
    )
    return ap.parse_args()


def _extract_subject_label(folder_name: str) -> str | None:
    # XY folders may already be BIDS-style subject folders: sub-<LABEL>
    m = re.match(r"^sub-([A-Za-z0-9]+)$", folder_name)
    if m:
        return m.group(1)

    # XY Psychopy folders may follow: XY_YYYYMMDD_NUM_CODE (CODE may include underscores).
    # Convert to the fMRIPrep participant label format: XYYYYYMMDDNUMCODE (no underscores).
    m = re.match(r"^XY_(\d{8})_(\d+)_([A-Za-z0-9_]+)$", folder_name)
    if m:
        date8, num, code = m.groups()
        code_clean = code.replace("_", "").upper()
        return f"XY{date8}{num}{code_clean}"

    # THU task_psych folders historically follow: THU_YYYYMMDD_NUM_CODE
    m = re.match(r"^THU_(\d{8})_(\d+)_([A-Za-z]+)$", folder_name)
    if not m:
        return None
    date8, num, code = m.groups()
    return f"THU{date8}{num}{code.upper()}"


def _has_task_files(subject_dir: Path) -> dict[str, bool]:
    flags = {"nback": False, "sst": False, "switch": False}
    for p in subject_dir.glob("*.csv"):
        name = p.name.lower()
        if "_nback_" in name:
            flags["nback"] = True
        elif "_sst_" in name:
            flags["sst"] = True
        elif "_switch_" in name:
            flags["switch"] = True
    return flags


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)
    external_inputs = dataset_cfg.get("external_inputs", {}) if isinstance(dataset_cfg, dict) else {}

    task_psych_dir = None
    if isinstance(external_inputs, dict):
        v = external_inputs.get("task_psych_dir")
        if v:
            p = Path(str(v))
            task_psych_dir = p if p.is_absolute() else (roots["raw_root"] / p)
    if task_psych_dir is None:
        task_psych_dir = roots["raw_root"] / "MRI_data" / "task_psych"

    if not task_psych_dir.exists():
        raise FileNotFoundError(f"task_psych_dir not found: {task_psych_dir}")

    out_path = Path(args.out) if args.out else (roots["processed_root"] / "table" / "sublist" / "taskfmri_sublist.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    counts = {"total": 0, "kept": 0, "missing_any": 0}
    for child in sorted(task_psych_dir.iterdir()):
        if not child.is_dir():
            continue
        label = _extract_subject_label(child.name)
        if label is None:
            continue
        counts["total"] += 1
        flags = _has_task_files(child)
        ok = True
        if args.require_all:
            ok = flags["nback"] and flags["sst"] and flags["switch"]
        else:
            ok = flags["nback"] or flags["sst"] or flags["switch"]
        if ok:
            labels.append(label)
            counts["kept"] += 1
        else:
            counts["missing_any"] += 1

    out_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    print(f"task_psych_dir={task_psych_dir}")
    print(f"out={out_path}")
    print(f"subjects_total={counts['total']}")
    print(f"subjects_written={counts['kept']}")
    print(f"subjects_excluded={counts['missing_any']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
