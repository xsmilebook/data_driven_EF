import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots

def efny_subject_to_thu_subject(subject_code: str) -> str:
    """
    EFNY_beh_metrics.csv:
      THU_20231014_131_ZXM_赵夕萌
    THU_app_results.csv:
      sub-THU20231014131ZXM
    """
    if subject_code is None or (isinstance(subject_code, float) and np.isnan(subject_code)):
        return np.nan
    s = str(subject_code).strip()

    # take first 4 underscore-separated tokens: THU, date, id, initials
    parts = s.split("_")
    if len(parts) >= 4 and parts[0].upper() == "THU":
        thu = parts[0]
        date = parts[1]
        sid = parts[2]
        init = parts[3]
        return f"sub-{thu}{date}{sid}{init}"

    # fallback regex
    m = re.search(r"(THU)[\s_]?(\d{8})[\s_]?(\d+)[\s_]?([A-Za-z]+)", s)
    if m:
        thu, date, sid, init = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"sub-{thu}{date}{sid}{init}"

    return np.nan


def first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _resolve_defaults(args) -> tuple[Path, Path, Path]:
    if not args.dataset:
        raise ValueError("Missing --dataset (required when defaults are used).")
    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    files_cfg = dataset_cfg.get("files", {})

    efny_rel = files_cfg.get("behavioral_metrics_file")
    thu_rel = files_cfg.get("thu_metrics_file")
    if not efny_rel:
        raise ValueError("Missing files.behavioral_metrics_file in dataset config.")
    if not thu_rel:
        raise ValueError("Missing files.thu_metrics_file in dataset config.")

    efny_csv = roots["processed_root"] / efny_rel
    thu_csv = roots["processed_root"] / thu_rel
    out_csv = roots["outputs_root"] / "results" / "behavior_data" / "efny_vs_thu_metric_compare.csv"
    return efny_csv, thu_csv, out_csv


def main():
    parser = argparse.ArgumentParser(description="Compare EFNY metrics with THU app results.")
    parser.add_argument("--efny-csv", default=None)
    parser.add_argument("--thu-csv", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    args = parser.parse_args()

    if args.efny_csv is None or args.thu_csv is None or args.out_csv is None:
        efny_csv, thu_csv, out_csv = _resolve_defaults(args)
        args.efny_csv = str(efny_csv)
        args.thu_csv = str(thu_csv)
        args.out_csv = str(out_csv)

    efny = pd.read_csv(args.efny_csv)
    thu = pd.read_csv(args.thu_csv, na_values=["NA", "NaN", "nan", ""])

    efny["subject"] = efny["subject_code"].map(efny_subject_to_thu_subject)
    efny = efny[efny["subject"].notna()].copy()

    merged = efny.merge(thu, on="subject", how="inner", suffixes=("_efny", "_thu"))
    if len(merged) == 0:
        raise RuntimeError("No matched subjects after merging. Check subject mapping rules.")

    # Explicit mapping: (metric_name, efny_col, thu_col)
    pairs = []

    def add(task, efny_col, thu_col):
        pairs.append((f"{task}:{thu_col}", efny_col, thu_col))

    # Conflict tasks
    for task in ["ColorStroop", "EmotionStroop", "FLANKER"]:
        add(task, f"{task}_Congruent_RT", f"{task}_cong_rt")
        add(task, f"{task}_Incongruent_RT", f"{task}_incong_rt")
        add(task, f"{task}_Contrast_RT", f"{task}_diff_rt")
        add(task, f"{task}_Congruent_ACC", f"{task}_cong_acc")
        add(task, f"{task}_Incongruent_ACC", f"{task}_incong_acc")
        add(task, f"{task}_Contrast_ACC", f"{task}_diff_acc")

    # CPT / GNG
    add("CPT", "CPT_Go_RT_Mean", "CPT_go_rt")
    add("CPT", "CPT_Go_ACC", "CPT_go_acc")
    add("CPT", "CPT_NoGo_ACC", "CPT_nogo_acc")

    add("GNG", "GNG_Go_RT_Mean", "GNG_go_rt")
    add("GNG", "GNG_Go_ACC", "GNG_go_acc")
    add("GNG", "GNG_NoGo_ACC", "GNG_nogo_acc")

    # Switch tasks (DCCS / DT / EmotionSwitch)
    for task in ["DCCS", "DT", "EmotionSwitch"]:
        add(task, f"{task}_Switch_RT", f"{task}_rt_switch")
        add(task, f"{task}_Repeat_RT", f"{task}_rt_repeat")
        add(task, f"{task}_Switch_ACC", f"{task}_acc_switch")
        add(task, f"{task}_Repeat_ACC", f"{task}_acc_repeat")
        add(task, f"{task}_Switch_Cost_RT", f"{task}_switch_cost_rt")
        add(task, f"{task}_Switch_Cost_ACC", f"{task}_switch_cost_acc")

    # SST
    add("SST", "SST_Go_RT_Mean", "SST_mean_go_rt")
    add("SST", "SST_Stop_ACC", "SST_stop_acc")
    add("SST", "SST_Mean_SSD", "SST_mean_ssd")
    add("SST", "SST_SSRT", "SST_ssrt")

    # FZSS (THU 用 fzss_* 命名)
    # mean_rt 这里优先用 Correct_RT_Mean，如果没有再退回 RT_Mean
    fzss_mean_rt = first_existing(merged, ["FZSS_Correct_RT_Mean", "FZSS_RT_Mean"])
    if fzss_mean_rt is None:
        fzss_mean_rt = "FZSS_Correct_RT_Mean"  # keep name; will be skipped if missing
    add("FZSS", "FZSS_Overall_ACC", "FZSS_fzss_acc")
    add("FZSS", "FZSS_Miss_Rate", "FZSS_fzss_miss_rate")
    add("FZSS", "FZSS_FA_Rate", "FZSS_fzss_fa_rate")
    pairs.append(("FZSS:FZSS_fzss_mean_rt", fzss_mean_rt, "FZSS_fzss_mean_rt"))

    # KT
    kt_rt = first_existing(merged, ["KT_Mean_RT", "KT_RT_Mean"])
    if kt_rt is None:
        kt_rt = "KT_Mean_RT"
    pairs.append(("KT:KT_mean_rt", kt_rt, "KT_mean_rt"))
    add("KT", "KT_Overall_ACC", "KT_acc")

    # N-back
    for task in ["oneback_number", "oneback_spatial", "oneback_emotion", "twoback_number", "twoback_spatial", "twoback_emotion"]:
        add(task, f"{task}_RT_Mean", f"{task}_mean_rt")
        add(task, f"{task}_ACC", f"{task}_acc")
        add(task, f"{task}_Hit_Rate", f"{task}_hit_rate")
        add(task, f"{task}_FA_Rate", f"{task}_fa_rate")
        add(task, f"{task}_dprime", f"{task}_dprime")

    # ZYST
    add("ZYST", "ZYST_T0_ACC", "ZYST_t0_acc")
    add("ZYST", "ZYST_T1_ACC", "ZYST_t1_acc")
    add("ZYST", "ZYST_T1_given_T0_ACC", "ZYST_t1_acc_given_t0_correct")

    rows = []
    for metric_name, c_efny, c_thu in pairs:
        if c_efny not in merged.columns or c_thu not in merged.columns:
            continue

        x = to_num(merged[c_efny])
        y = to_num(merged[c_thu])
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n == 0:
            continue

        xr = x[m]
        yr = y[m]

        pearson = xr.corr(yr, method="pearson") if n >= 2 else np.nan
        spearman = xr.corr(yr, method="spearman") if n >= 2 else np.nan

        diff = (xr - yr).abs()
        idx = diff.idxmax()
        max_abs_diff = float(diff.loc[idx]) if n > 0 else np.nan
        subj_max = merged.loc[idx, "subject"]
        v_efny = float(x.loc[idx]) if pd.notna(x.loc[idx]) else np.nan
        v_thu = float(y.loc[idx]) if pd.notna(y.loc[idx]) else np.nan

        rows.append(
            {
                "metric": metric_name,
                "efny_col": c_efny,
                "thu_col": c_thu,
                "n": n,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": max_abs_diff,
                "subject_of_max_abs_diff": subj_max,
                "efny_value_at_max": v_efny,
                "thu_value_at_max": v_thu,
            }
        )

    out = pd.DataFrame(rows).sort_values(["pearson_r", "max_abs_diff"], ascending=[True, False])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
