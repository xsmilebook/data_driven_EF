import argparse
import csv
import re
import sys
from pathlib import Path
import numpy as np

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots

# Example:
# python -m src.imaging_preprocess.screen_head_motion_efny --dataset EFNY --config configs/paths.yaml

def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def parse_task_run(name: str) -> str:
    task_match = re.search(r"task-([A-Za-z0-9]+)", name)
    run_match = re.search(r"run-([0-9]+)", name)
    task = task_match.group(1) if task_match else ""
    run = run_match.group(1) if run_match else ""
    if task and run:
        return f"{task}_{run}"
    return task


def summarize_fd(tsv_path: Path) -> tuple[int, str, int, str]:
    frame_count = 0
    total = 0.0
    valid = 0
    high = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return 0, "NA", 0, "NA"
        for row in reader:
            frame_count += 1
            v = row.get("framewise_displacement")
            if v is None:
                continue
            s = v.strip()
            if not s or s.lower() == "n/a":
                continue
            try:
                x = float(s)
            except ValueError:
                continue
            total += x
            valid += 1
            if x > 0.3:
                high += 1
    if valid == 0:
        return frame_count, "NA", 0, "NA"
    mean = total / valid
    ratio = high / valid
    return frame_count, f"{mean:.6f}", valid, f"{ratio:.6f}"

def extract_fd_values(tsv_path: Path) -> list[float]:
    vals = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return vals
        for row in reader:
            v = row.get("framewise_displacement")
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() == "n/a":
                continue
            try:
                x = float(s)
            except ValueError:
                continue
            vals.append(x)
    return vals


def collect_rows(fmriprep_dir: Path) -> list[dict]:
    subjects = {}
    global_vals = []
    for tsv in fmriprep_dir.rglob("*task-rest*desc-confounds_timeseries.tsv"):
        if "func" not in tsv.parts:
            continue
        subject_id = find_subject_id(tsv)
        m = re.search(r"run-([0-9]+)", tsv.name)
        if not m:
            continue
        run_idx = int(m.group(1))
        if run_idx < 1 or run_idx > 4:
            continue
        frame_num, mean_fd, valid_count, high_ratio = summarize_fd(tsv)
        if frame_num == 180:
            vals = extract_fd_values(tsv)
            if len(vals) > 0:
                global_vals.extend(vals)
        runs = subjects.setdefault(subject_id, {})
        runs[run_idx] = {
            "frame": str(frame_num),
            "fd": mean_fd,
            "high_ratio": high_ratio,
        }
    upper_limit = float("inf")
    if len(global_vals) > 0:
        q1 = float(np.percentile(global_vals, 25))
        q3 = float(np.percentile(global_vals, 75))
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
    rows = []
    for subject_id in sorted(subjects.keys()):
        runs = subjects[subject_id]
        row = {"subid": subject_id}
        valid_num = 0
        valid_fd_values = []  # Store FD values for valid sessions
        
        for i in range(1, 5):
            r = runs.get(i)
            if r:
                row[f"rest{i}_frame"] = r["frame"]
                row[f"rest{i}_fd"] = r["fd"]
                hr = r.get("high_ratio", "NA")
                row[f"rest{i}_high_ratio"] = hr
                is_valid = False
                try:
                    is_valid = (row[f"rest{i}_frame"] == "180") and (hr != "NA" and float(hr) <= 0.25) and (r["fd"] != "NA" and float(r["fd"]) <= upper_limit)
                except Exception:
                    is_valid = False
                row[f"rest{i}_valid"] = "1" if is_valid else "0"
                if is_valid:
                    valid_num += 1
                    try:
                        # Add FD value to list for mean calculation
                        fd_value = float(r["fd"])
                        valid_fd_values.append(fd_value)
                    except (ValueError, TypeError):
                        pass  # Skip invalid FD values
            else:
                row[f"rest{i}_frame"] = "NA"
                row[f"rest{i}_fd"] = "NA"
                row[f"rest{i}_high_ratio"] = "NA"
                row[f"rest{i}_valid"] = "0"
        
        # Calculate mean FD for valid sessions
        if valid_fd_values:
            mean_fd = sum(valid_fd_values) / len(valid_fd_values)
            row["meanFD"] = f"{mean_fd:.6f}"
        else:
            row["meanFD"] = "NA"
            
        row["valid_num"] = str(valid_num)
        row["valid_subject"] = "1" if valid_num >= 2 else "0"
        rows.append(row)
    rows.sort(key=lambda r: r["subid"])
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "subid",
        "rest1_frame", "rest1_fd", "rest1_high_ratio", "rest1_valid",
        "rest2_frame", "rest2_fd", "rest2_high_ratio", "rest2_valid",
        "rest3_frame", "rest3_fd", "rest3_high_ratio", "rest3_valid",
        "rest4_frame", "rest4_fd", "rest4_high_ratio", "rest4_valid",
        "valid_num", "valid_subject", "meanFD",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _resolve_defaults(args) -> tuple[Path, Path]:
    if not args.dataset:
        raise ValueError("Missing --dataset (required when --fmriprep-dir is not provided).")
    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    external_inputs = dataset_cfg.get("external_inputs", {})
    fmriprep_dir = ""
    if isinstance(external_inputs, dict):
        fmriprep_dir = str(external_inputs.get("fmriprep_dir", "") or "")
    if not fmriprep_dir:
        raise ValueError("Missing fmriprep_dir in dataset config and no --fmriprep-dir provided.")

    out_default = roots["interim_root"] / "table" / "qc" / "rest_fd_summary.csv"
    return Path(fmriprep_dir), out_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    args = parser.parse_args()

    if args.fmriprep_dir:
        fdir = Path(args.fmriprep_dir)
        out_path = Path(args.out) if args.out else Path("rest_fd_summary.csv")
    else:
        fdir, out_default = _resolve_defaults(args)
        out_path = Path(args.out) if args.out else out_default

    if not fdir.exists():
        print(f"Input directory not found: {fdir}", file=sys.stderr)
        return
    rows = collect_rows(fdir)
    write_csv(rows, out_path)
    eligible = sum(1 for r in rows if r.get("valid_subject") == "1")
    excluded = len(rows) - eligible
    print(f"excluded={excluded}")
    print(f"eligible={eligible}")


if __name__ == "__main__":
    main()
