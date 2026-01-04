from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def _iter_run_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/run.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rows(ssm_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_path in _iter_run_files(ssm_root):
        run_dir = run_path.parent
        run = _read_json(run_path)
        loo_path = run_dir / "loo.json"
        qc_path = run_dir / "qc.json"
        loo = _read_json(loo_path) if loo_path.exists() else {}
        qc = _read_json(qc_path) if qc_path.exists() else {}
        rows.append(
            {
                "dataset": run.get("dataset"),
                "task": run.get("task"),
                "model": run.get("model"),
                "model_id": run.get("model_id"),
                "dataset_kind": run.get("dataset_kind", ""),
                "n_trials": run.get("n_trials"),
                "n_subjects": run.get("n_subjects"),
                "elpd_loo": loo.get("elpd_loo"),
                "p_loo": loo.get("p_loo"),
                "path": str(run_dir),
                "qc": json.dumps(qc, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _add_pairwise_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple within-task Δelpd_loo for pre-defined null/effect pairs.

    Notes:
    - Only comparable when fitted on the same dataset (same task and dataset_kind).
    - Mixing vs Switch models are not directly comparable because they use different trial subsets.
    """

    pairs = [
        ("FLANKER", "flanker_null", "flanker_congruency_v"),
        ("ColorStroop", "colorstroop_race4_null", "colorstroop_race4_congruency_v"),
        ("EmotionStroop", "emotionstroop_race4_null", "emotionstroop_race4_congruency_v"),
        ("DT", "dt_mixing_null", "dt_mixing"),
        ("DT", "dt_switch_null", "dt_switch"),
        ("EmotionSwitch", "emotionswitch_mixing_null", "emotionswitch_mixing"),
        ("EmotionSwitch", "emotionswitch_switch_null", "emotionswitch_switch"),
    ]

    delta_rows: list[dict] = []
    for task, null_id, eff_id in pairs:
        a = df[(df["task"] == task) & (df["model_id"] == null_id)]
        b = df[(df["task"] == task) & (df["model_id"] == eff_id)]
        if a.empty or b.empty:
            continue
        # align by dataset_kind to avoid mixing mixed-only and all-trials
        for kind in sorted(set(a["dataset_kind"]).intersection(set(b["dataset_kind"]))):
            a1 = a[a["dataset_kind"] == kind].iloc[0]
            b1 = b[b["dataset_kind"] == kind].iloc[0]
            if pd.isna(a1.get("elpd_loo")) or pd.isna(b1.get("elpd_loo")):
                continue
            delta_rows.append(
                {
                    "task": task,
                    "dataset_kind": kind,
                    "null_model_id": null_id,
                    "effect_model_id": eff_id,
                    "delta_elpd_loo": float(b1["elpd_loo"]) - float(a1["elpd_loo"]),
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        return df
    df = df.copy()
    df["delta_elpd_loo"] = pd.NA
    # write deltas onto the effect-model rows
    for _, r in delta_df.iterrows():
        mask = (df["task"] == r["task"]) & (df["model_id"] == r["effect_model_id"]) & (df["dataset_kind"] == r["dataset_kind"])
        df.loc[mask, "delta_elpd_loo"] = r["delta_elpd_loo"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect saved SSM run metadata into a single CSV table.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--out", type=str, default=None, help="Default: processed/table/metrics/ssm/<dataset>/ssm_summary.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=None, repo_root=repo_root)
    _ = dataset_cfg  # reserved for future dataset-specific overrides

    ssm_root = roots["processed_root"] / "table" / "metrics" / "ssm" / args.dataset
    df = _collect_rows(ssm_root)
    if df.empty:
        raise SystemExit(f"No run.json found under: {ssm_root}")
    df = _add_pairwise_deltas(df)

    out_path = Path(args.out) if args.out else (ssm_root / "ssm_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["task", "model_id"]).to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

