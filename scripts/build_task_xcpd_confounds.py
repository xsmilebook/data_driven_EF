#!/usr/bin/env python3
"""
Build a BIDS-derivatives dataset of task regressors for XCP-D task regression (block HRF + event FIR).

For xcp_d >= 0.10.0, the recommended pattern is:
- `--datasets custom=/path/to/custom_confounds`
- `--nuisance-regressors /path/to/custom_config.yml`

where `/path/to/custom_confounds` is a valid BIDS-derivatives dataset (has `dataset_description.json`),
and contains per-subject confounds TSVs whose basenames match the fMRIPrep confounds TSVs.

This script reads:
- Psychopy behavior CSVs under task_psych_dir (e.g., data/raw/MRI_data/task_psych/)
- fMRIPrep confounds TSV + BOLD JSON (for N volumes and TR)

And writes:
- <out_root>/dataset_description.json
- <out_root>/confounds_config.yml
- <out_root>/sub-<label>/func/<...desc-confounds_timeseries.tsv> (task regressor matrix; basename matches fMRIPrep confounds TSV)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import glob
import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


_CONF_36P_COLUMNS = [
    "trans_x",
    "trans_x_derivative1",
    "trans_x_derivative1_power2",
    "trans_x_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_derivative1_power2",
    "trans_y_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_derivative1_power2",
    "trans_z_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_power2",
    "rot_x_derivative1_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_power2",
    "rot_y_derivative1_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_power2",
    "rot_z_derivative1_power2",
    "global_signal",
    "global_signal_derivative1",
    "global_signal_power2",
    "global_signal_derivative1_power2",
    "csf",
    "csf_derivative1",
    "csf_power2",
    "csf_derivative1_power2",
    "white_matter",
    "white_matter_derivative1",
    "white_matter_power2",
    "white_matter_derivative1_power2",
]


@dataclass(frozen=True)
class Event:
    onset: float
    duration: float
    trial_type: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="BIDS participant label, with or without 'sub-' prefix.")
    ap.add_argument("--task", required=True, choices=["nback", "sst", "switch"])
    ap.add_argument("--out-root", required=True, help="Output root for custom confounds BIDS-derivatives dataset.")
    ap.add_argument("--behavior-file", default=None, help="Optional explicit Psychopy CSV path.")
    ap.add_argument("--fmriprep-dir", default=None, help="Optional explicit fMRIPrep derivatives root.")
    ap.add_argument("--task-psych-dir", default=None, help="Optional explicit task_psych directory.")
    ap.add_argument("--fir-window-seconds", type=float, default=20.0)
    ap.add_argument("--oversampling", type=int, default=16)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    return ap.parse_args()


def _strip_sub_prefix(subject: str) -> str:
    s = subject.strip()
    return s[4:] if s.lower().startswith("sub-") else s


def _float_or_none(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "nan", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(v: str | None) -> int | None:
    x = _float_or_none(v)
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _read_psychopy_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


_PSYCHOPY_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})\.(\d{2})\.(\d{3})(?=\.|$)")


def _psychopy_timestamp_from_filename(name: str) -> datetime | None:
    """Parse Psychopy timestamp from filename (YYYY-MM-DD_HHhMM.SS.mmm)."""
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


def _choose_newest_by_filename_timestamp(paths: list[Path]) -> Path:
    """Choose newest CSV by filename timestamp; fall back to mtime if missing."""

    def key(p: Path) -> tuple[int, datetime, float]:
        ts = _psychopy_timestamp_from_filename(p.name)
        if ts is None:
            return (0, datetime.min, p.stat().st_mtime)
        return (1, ts, p.stat().st_mtime)

    return sorted(set(paths), key=key, reverse=True)[0]

def _extract_subject_label_from_task_psych_dirname(folder_name: str) -> str | None:
    m = re.match(r"^sub-([A-Za-z0-9]+)$", folder_name)
    if m:
        return m.group(1)

    m = re.match(r"^THU_(\d{8})_(\d+)_([A-Za-z]+)$", folder_name)
    if m:
        date8, num, code = m.groups()
        return f"THU{date8}{num}{code.upper()}"

    # XY Psychopy folders: XY_YYYYMMDD_NUM_CODE (CODE may include underscores).
    m = re.match(r"^XY_(\d{8})_(\d+)_([A-Za-z0-9_]+)$", folder_name)
    if m:
        date8, num, code = m.groups()
        return f"XY{date8}{num}{code.replace('_', '').upper()}"

    return None


def _find_behavior_file(task_psych_dir: Path, subject_label: str, task: str) -> Path:
    label = _strip_sub_prefix(subject_label)

    # Prefer an exact match against a subject folder name when possible.
    subj_dir: Path | None = None
    for child in task_psych_dir.iterdir():
        if not child.is_dir():
            continue
        extracted = _extract_subject_label_from_task_psych_dirname(child.name)
        if extracted and extracted == label:
            subj_dir = child
            break
    if subj_dir is not None:
        candidates: list[Path] = []
        for p in subj_dir.glob("*.csv"):
            name = p.name.lower()
            if task == "sst":
                ok = "_sst_" in name
            else:
                ok = f"_{task}_" in name
            if ok:
                candidates.append(p)
        if candidates:
            return _choose_newest_by_filename_timestamp(candidates)

    m = re.match(r"^(THU)(\d{8})(\d+)([A-Za-z]+)$", label)
    patterns: list[str] = []
    if m:
        _, date8, num, code = m.groups()
        prefix = f"THU_{date8}_{num}_{code}"
        task_variants = [task, task.upper()]
        if task == "sst":
            task_variants = ["SST", "sst"]
        for tv in task_variants:
            patterns.append(str(task_psych_dir / "**" / prefix / f"{prefix}_{tv}_*.csv"))
            patterns.append(str(task_psych_dir / "**" / f"{prefix}_{tv}_*.csv"))
    else:
        patterns.append(str(task_psych_dir / "**" / f"*{label}*{task}*.csv"))
        patterns.append(str(task_psych_dir / "**" / f"*{label}*{task.upper()}*.csv"))

    matches: list[Path] = []
    for pat in patterns:
        matches.extend(Path(p) for p in glob.glob(pat, recursive=True))

    if not matches:
        for p in task_psych_dir.rglob("*.csv"):
            name = p.name.lower()
            if label.lower() in name and task.lower() in name:
                matches.append(p)

    if not matches:
        raise FileNotFoundError(f"No behavior CSV found for subject={label}, task={task} under {task_psych_dir}")
    return _choose_newest_by_filename_timestamp(matches)


def _resolve_defaults(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if not args.dataset:
        return None, None
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    external_inputs = dataset_cfg.get("external_inputs", {})
    if not isinstance(external_inputs, dict):
        return None, None

    task_psych = external_inputs.get("task_psych_dir")
    task_psych_dir = None
    if task_psych:
        p = Path(str(task_psych))
        task_psych_dir = p if p.is_absolute() else (roots["raw_root"] / p)

    fmriprep_dir = None
    task_dirs = external_inputs.get("fmriprep_task_dirs", {})
    if isinstance(task_dirs, dict) and args.task in task_dirs:
        fmriprep_dir = Path(str(task_dirs[args.task]))
    else:
        k = f"fmriprep_task_{args.task}_dir"
        v = external_inputs.get(k)
        if v:
            fmriprep_dir = Path(str(v))
    if fmriprep_dir is not None and not fmriprep_dir.is_absolute():
        fmriprep_dir = (roots["repo_root"] / fmriprep_dir).resolve()

    return fmriprep_dir, task_psych_dir


def _find_confounds_files(fmriprep_dir: Path, subject_label: str, task: str) -> list[Path]:
    label = _strip_sub_prefix(subject_label)
    sub_token = f"sub-{label}"
    matches: list[Path] = []
    for p in fmriprep_dir.rglob(f"*task-{task}*desc-confounds_timeseries.tsv"):
        if sub_token in p.name and "func" in p.parts:
            matches.append(p)
    return sorted(matches)


def _count_tsv_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for i, _ in enumerate(f):
            n = i
    return max(0, n)


def _load_tr_from_bold_json(confounds_tsv: Path) -> float:
    func_dir = confounds_tsv.parent
    stem = confounds_tsv.name.replace("_desc-confounds_timeseries.tsv", "")
    candidates = sorted(func_dir.glob(stem + "*desc-preproc_bold.json"))
    if not candidates:
        candidates = sorted(func_dir.glob(stem + "*bold.json"))
    if not candidates:
        raise FileNotFoundError(f"No BOLD JSON found near confounds file: {confounds_tsv}")
    with candidates[0].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    tr = meta.get("RepetitionTime")
    if tr is None:
        raise ValueError(f"Missing RepetitionTime in {candidates[0]}")
    return float(tr)


def _spm_hrf(dt: float, *, time_length: float = 32.0) -> np.ndarray:
    t = np.arange(0, time_length, dt)
    # Parameters align with SPM canonical HRF defaults.
    peak1, peak2 = 6.0, 16.0
    dispersion1, dispersion2 = 1.0, 1.0
    under_ratio = 6.0

    def _gamma_pdf(x: np.ndarray, *, shape: float, scale: float) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        m = x > 0
        xm = x[m]
        y[m] = (xm ** (shape - 1)) * np.exp(-xm / scale) / (math.gamma(shape) * (scale**shape))
        return y

    h = _gamma_pdf(t, shape=peak1 / dispersion1, scale=dispersion1) - _gamma_pdf(
        t, shape=peak2 / dispersion2, scale=dispersion2
    ) / under_ratio
    s = float(np.abs(h).sum())
    return h / s if s > 0 else h


def _boxcar_highres(
    timepoints: np.ndarray,
    events: Iterable[Event],
    trial_type: str,
) -> np.ndarray:
    x = np.zeros_like(timepoints, dtype=float)
    for e in events:
        if e.trial_type != trial_type:
            continue
        onset = max(0.0, float(e.onset))
        offset = max(onset, float(e.onset + e.duration))
        m = (timepoints >= onset) & (timepoints < offset)
        x[m] = 1.0
    return x


def _convolve_and_downsample(
    x: np.ndarray,
    h: np.ndarray,
    *,
    n_volumes: int,
    oversampling: int,
) -> np.ndarray:
    y = np.convolve(x, h, mode="full")[: x.shape[0]]
    expected = n_volumes * oversampling
    if y.shape[0] < expected:
        y = np.pad(y, (0, expected - y.shape[0]))
    y = y[:expected]
    y = y.reshape((n_volumes, oversampling)).mean(axis=1)
    return y


def _fir_matrix(
    n_volumes: int,
    tr: float,
    events: Iterable[Event],
    trial_type: str,
    delays: list[int],
) -> dict[str, np.ndarray]:
    counts = np.zeros(n_volumes, dtype=float)
    for e in events:
        if e.trial_type != trial_type:
            continue
        if e.onset < 0:
            continue
        bin_idx = int(np.floor(e.onset / tr))
        if 0 <= bin_idx < n_volumes:
            counts[bin_idx] += 1.0
    cols: dict[str, np.ndarray] = {}
    for d in delays:
        v = np.zeros(n_volumes, dtype=float)
        if d == 0:
            v[:] = counts
        else:
            v[d:] = counts[:-d]
        cols[f"{trial_type}_delay_{d}"] = v
    return cols


def _task_expected_block_types(task: str) -> list[str]:
    if task == "nback":
        return ["state_pure_0back", "state_pure_2back", "state_mixed"]
    if task == "switch":
        return ["state_pure_red", "state_pure_blue", "state_mixed"]
    if task == "sst":
        return ["state_part1", "state_part2"]
    raise ValueError(task)

def _block_types_present(task: str, block_events: Iterable[Event]) -> list[str]:
    present = {e.trial_type for e in block_events}
    order = _task_expected_block_types(task)
    return [t for t in order if t in present]


def _task_event_types(task: str) -> list[str]:
    if task == "sst":
        return ["stimulus", "banana", "response"]
    return ["stimulus", "response"]


def _detect_t0(rows: list[dict[str, str]]) -> float:
    for r in rows:
        v = _float_or_none(r.get("MRI_Signal_s.started"))
        if v is not None:
            return float(v)
    for r in rows:
        v = _float_or_none(r.get("Begin_fix.started"))
        if v is not None:
            return float(v)
    return 0.0


def _build_events_from_psychopy(rows: list[dict[str, str]], task: str) -> tuple[list[Event], list[Event]]:
    t0 = _detect_t0(rows)

    def _row_time(r: dict[str, str], keys: list[str]) -> float | None:
        for k in keys:
            v = _float_or_none(r.get(k))
            if v is not None:
                return float(v)
        return None

    # IMPORTANT: Trial_fix is a fixation period, while Trial_text/Trial_image_1 is stimulus onset.
    # For event regressors we always use stimulus onset columns.
    # For block/state regressors, we prefer stimulus timing for block start/stop to avoid shifting
    # the canonical HRF regressors earlier by the fixation duration.
    if task in {"nback", "switch"}:
        trial_start_keys = ["Trial_text.started", "Trial.started", "Trial_fix.started", "key_resp.started"]
        trial_stop_keys = ["Trial_text.stopped", "Trial.stopped", "key_resp.stopped", "Trial_fix.stopped"]
        trial_row_keys = ["Trial_text.started", "Trial.started", "Trial_fix.started"]
    elif task == "sst":
        trial_start_keys = ["Trial_image_1.started", "Trial.started", "Trial_fix.started", "key_resp.started"]
        trial_stop_keys = ["Trial_image_1.stopped", "Trial.stopped", "key_resp.stopped", "Trial_fix.stopped"]
        trial_row_keys = ["Trial_image_1.started", "Trial.started", "Trial_fix.started"]
    else:
        raise ValueError(task)

    trial_rows = [r for r in rows if any(_float_or_none(r.get(k)) is not None for k in trial_row_keys)]
    if not trial_rows:
        raise ValueError(
            "No trial rows found (missing expected trial timing columns). "
            f"task={task} expected_any_of={trial_row_keys}"
        )

    block_events: list[Event] = []
    event_events: list[Event] = []

    if task in {"nback", "switch"}:
        blocks: dict[int, list[dict[str, str]]] = {}
        for r in trial_rows:
            idx = _int_or_none(r.get("Task_loop.thisIndex"))
            if idx is None:
                idx = 0
            blocks.setdefault(idx, []).append(r)

        for _, trs in sorted(blocks.items(), key=lambda kv: kv[0]):
            starts = [t for t in (_row_time(r, trial_start_keys) for r in trs) if t is not None]
            stops = [t for t in (_row_time(r, trial_stop_keys) for r in trs) if t is not None]
            if not starts:
                continue
            trial_start = min(starts)
            trial_stop = max(stops) if stops else trial_start
            block_hint = ""
            for r in trs:
                v = (r.get("Task_img") or "").strip()
                if v:
                    block_hint = v.lower()
                    break
                v = (r.get("Trial_loop_list") or "").strip()
                if v:
                    block_hint = v.lower()
                    break
            if task == "nback":
                if "0back" in block_hint:
                    label = "state_pure_0back"
                elif "2back" in block_hint:
                    label = "state_pure_2back"
                else:
                    label = "state_mixed"
            else:  # switch
                if "nonswitch1" in block_hint:
                    label = "state_pure_red"
                elif "nonswitch2" in block_hint:
                    label = "state_pure_blue"
                else:
                    label = "state_mixed"
            onset = float(trial_start) - t0
            duration = max(0.0, float(trial_stop) - float(trial_start))
            block_events.append(Event(onset=onset, duration=duration, trial_type=label))

        stim_col = "Trial_text.started"
        for r in trial_rows:
            stim = _float_or_none(r.get(stim_col))
            if stim is not None:
                event_events.append(Event(onset=float(stim) - t0, duration=0.0, trial_type="stimulus"))
            ks = _float_or_none(r.get("key_resp.started"))
            rt = _float_or_none(r.get("key_resp.rt"))
            if ks is not None and rt is not None:
                event_events.append(Event(onset=float(ks + rt) - t0, duration=0.0, trial_type="response"))

    elif task == "sst":
        # Some subjects have 120 trials (single block).
        # Some subjects have 180 trials (two blocks), with an ~15s fixation gap between trial 90 and 91.
        # When available, use Trial_loop_list to identify loop1/loop2 (each loop may include an extra row with no stimulus).
        loop_key = "Trial_loop_list"
        groups: list[tuple[str, list[dict[str, str]]]] = []
        if loop_key in trial_rows[0]:
            by_loop: dict[str, list[dict[str, str]]] = {}
            for r in trial_rows:
                v = (r.get(loop_key) or "").strip()
                if v:
                    by_loop.setdefault(v, []).append(r)
            if len(by_loop) >= 2:
                for k, trs in by_loop.items():
                    groups.append((k, trs))
                groups.sort(key=lambda kv: min(t for t in (_row_time(r, trial_start_keys) for r in kv[1]) if t is not None))

        if not groups:
            # Fallback: infer by trial count. If >= 180 trials, split into two blocks at TrialN 90.
            tns = [_int_or_none(r.get("Trial_loop.thisTrialN")) for r in trial_rows]
            tns = [t for t in tns if t is not None]
            is_two_block = False
            if tns:
                is_two_block = (max(tns) >= 179) or (len(set(tns)) >= 170)
            if is_two_block:
                b1, b2 = [], []
                for r in trial_rows:
                    tn = _int_or_none(r.get("Trial_loop.thisTrialN"))
                    if tn is None:
                        continue
                    (b1 if tn < 90 else b2).append(r)
                if b1:
                    groups.append(("block1", b1))
                if b2:
                    groups.append(("block2", b2))
            else:
                groups.append(("block1", trial_rows))

        # Map to fixed labels for downstream consistency.
        label_map = {}
        if len(groups) == 1:
            label_map[groups[0][0]] = "state_part1"
        else:
            label_map[groups[0][0]] = "state_part1"
            label_map[groups[1][0]] = "state_part2"

        for _, trs in groups[:2]:
            label = label_map[_]
            starts = [t for t in (_row_time(r, trial_start_keys) for r in trs) if t is not None]
            stops = [t for t in (_row_time(r, trial_stop_keys) for r in trs) if t is not None]
            if not starts:
                continue
            trial_start = min(starts)
            trial_stop = max(stops) if stops else trial_start
            onset = float(trial_start) - t0
            duration = max(0.0, float(trial_stop) - float(trial_start))
            block_events.append(Event(onset=onset, duration=duration, trial_type=label))

        for r in trial_rows:
            stim = _float_or_none(r.get("Trial_image_1.started"))
            if stim is not None:
                event_events.append(Event(onset=float(stim) - t0, duration=0.0, trial_type="stimulus"))
            bad = (r.get("bad") or "").lower()
            if "banana" in bad:
                banana = _float_or_none(r.get("Trial_image_3.started"))
                if banana is not None:
                    event_events.append(Event(onset=float(banana) - t0, duration=0.0, trial_type="banana"))
            ks = _float_or_none(r.get("key_resp.started"))
            rt = _float_or_none(r.get("key_resp.rt"))
            if ks is not None and rt is not None:
                event_events.append(Event(onset=float(ks + rt) - t0, duration=0.0, trial_type="response"))
    else:
        raise ValueError(task)

    return block_events, event_events


def _write_tsv(path: Path, columns: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(columns)
        for row in matrix:
            w.writerow([f"{x:.8f}" for x in row.tolist()])


def _write_dataset_description(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "dataset_description.json"
    if path.exists():
        return
    obj = {
        "Name": "Custom Confounds (Task Regression)",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "data_driven_EF/scripts.build_task_xcpd_confounds"}],
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_confounds_config(path: Path, task_columns: list[str]) -> None:
    # XCP-D YAML format: each confound group specifies a dataset and a BIDS query, plus columns.
    yml = [
        "name: 36P_plus_task",
        "description: |",
        "  36P nuisance regressors plus task regressors (block HRF + event FIR).",
        "confounds:",
        "  preproc_confounds:",
        "    dataset: preprocessed",
        "    query:",
        "      space: null",
        "      cohort: null",
        "      res: null",
        "      den: null",
        "      desc: confounds",
        "      extension: .tsv",
        "      suffix: timeseries",
        "    columns:",
    ]
    for c in _CONF_36P_COLUMNS:
        yml.append(f"    - {c}")
    yml.extend(
        [
            "  task_confounds:",
            "    dataset: custom",
            "    query:",
            "      space: null",
            "      cohort: null",
            "      res: null",
            "      den: null",
            "      desc: confounds",
            "      extension: .tsv",
            "      suffix: timeseries",
            "    columns:",
        ]
    )
    for c in task_columns:
        yml.append(f"    - {c}")
    path.write_text("\n".join(yml) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    subject = _strip_sub_prefix(args.subject)
    out_root = Path(args.out_root).resolve()

    fmriprep_dir = Path(args.fmriprep_dir).resolve() if args.fmriprep_dir else None
    task_psych_dir = Path(args.task_psych_dir).resolve() if args.task_psych_dir else None
    if fmriprep_dir is None or task_psych_dir is None:
        inferred_fmriprep, inferred_task_psych = _resolve_defaults(args)
        fmriprep_dir = fmriprep_dir or inferred_fmriprep
        task_psych_dir = task_psych_dir or inferred_task_psych

    if fmriprep_dir is None or not fmriprep_dir.exists():
        raise FileNotFoundError("Missing/invalid fmriprep dir (set --fmriprep-dir or dataset config external_inputs).")
    if task_psych_dir is None or not task_psych_dir.exists():
        raise FileNotFoundError("Missing/invalid task_psych dir (set --task-psych-dir or dataset config external_inputs).")

    behavior_file = Path(args.behavior_file).resolve() if args.behavior_file else _find_behavior_file(task_psych_dir, subject, args.task)
    rows = _read_psychopy_rows(behavior_file)
    block_events, event_events = _build_events_from_psychopy(rows, args.task)

    confounds_files = _find_confounds_files(fmriprep_dir, subject, args.task)
    if len(confounds_files) == 0:
        raise FileNotFoundError(f"No fMRIPrep confounds found for subject={subject}, task={args.task} in {fmriprep_dir}")
    if len(confounds_files) > 1:
        raise RuntimeError(
            f"Found multiple fMRIPrep confounds files for subject={subject}, task={args.task}. "
            "Pass --behavior-file and run per-run explicitly or reduce to a single run."
        )
    confounds_tsv = confounds_files[0]
    n_volumes = _count_tsv_rows(confounds_tsv)
    tr = _load_tr_from_bold_json(confounds_tsv)

    fir_delays = list(range(0, int(max(0.0, round(args.fir_window_seconds / tr))) + 1))
    oversampling = max(1, int(args.oversampling))
    dt = float(tr) / oversampling
    timepoints = np.arange(0, n_volumes * float(tr), dt, dtype=float)
    hrf = _spm_hrf(dt)

    block_types = _block_types_present(args.task, block_events)
    event_types = _task_event_types(args.task)
    task_columns: list[str] = []
    cols: dict[str, np.ndarray] = {}

    for bt in block_types:
        x = _boxcar_highres(timepoints, block_events, bt)
        cols[bt] = _convolve_and_downsample(x, hrf, n_volumes=n_volumes, oversampling=oversampling)
        task_columns.append(bt)

    for et in event_types:
        for k, v in _fir_matrix(n_volumes, tr, event_events, et, fir_delays).items():
            cols[k] = v
            task_columns.append(k)

    mat = np.column_stack([cols[c] for c in task_columns]).astype(float)
    mat[~np.isfinite(mat)] = 0.0

    # Drop all-zero columns (e.g., single-block SST may not have state_part2;
    # banana FIR columns may be all zero for subjects with no banana events).
    keep_idx = [i for i in range(mat.shape[1]) if float(np.abs(mat[:, i]).sum()) > 0.0]
    if keep_idx and len(keep_idx) < mat.shape[1]:
        mat = mat[:, keep_idx]
        task_columns = [task_columns[i] for i in keep_idx]

    _write_dataset_description(out_root)
    out_tsv = out_root / f"sub-{subject}" / "func" / confounds_tsv.name
    _write_tsv(out_tsv, task_columns, mat)
    _write_confounds_config(out_root / "confounds_config.yml", task_columns)

    print(f"behavior_file={behavior_file}")
    print(f"confounds_tsv={confounds_tsv}")
    print(f"TR={tr}")
    print(f"n_volumes={n_volumes}")
    print(f"custom_confounds_root={out_root}")
    print(f"custom_confounds_tsv={out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
