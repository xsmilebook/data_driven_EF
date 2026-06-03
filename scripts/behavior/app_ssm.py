from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.behavior._common import add_common_arguments, load_runtime_config
from src.behavior.app.ssm import (
    available_model_names,
    build_ssm_trials,
    fit_ssm_model,
    get_model_spec,
    get_ssm_config,
    p_outlier_setting,
    sampling_config,
    summarize_ssm,
    trial_qc,
    write_trace,
)
from src.common import write_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit THU app SSM/DDM models.")
    add_common_arguments(parser)
    parser.add_argument("--model", default="all", help="SSM model name or 'all'.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--tune", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _select_subjects(frame: pd.DataFrame, max_subjects: int | None) -> pd.DataFrame:
    if max_subjects is None:
        return frame
    subjects = sorted(frame["subject_code"].dropna().unique())[:max_subjects]
    return frame.loc[frame["subject_code"].isin(subjects)].copy()


def _empty_summary(model_name: str, status: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model_name": model_name, "status": status, "summary_reason": reason}]
    )


def _sampling_from_args(config: dict, args: argparse.Namespace) -> dict:
    sampling = sampling_config(config, args.mode)
    for key in ("draws", "tune", "chains", "seed"):
        value = getattr(args, key)
        if value is not None:
            sampling[key] = value
    return sampling


def _fit_one_model(
    trials: pd.DataFrame,
    config: dict,
    model_name: str,
    ssm_dir: Path,
    args: argparse.Namespace,
) -> None:
    spec = get_model_spec(config, model_name)
    model_trials = build_ssm_trials(trials, config, model_name)
    model_trials = _select_subjects(model_trials, args.max_subjects)
    qc = trial_qc(model_trials, spec)

    write_csv(model_trials, ssm_dir / f"{model_name}_trials.csv")
    write_csv(qc, ssm_dir / f"{model_name}_qc.csv")

    first_status = str(qc.iloc[0]["status"]) if not qc.empty else "trial_qc_fail"
    if first_status == "trial_qc_fail":
        write_csv(
            _empty_summary(model_name, "trial_qc_fail", str(qc.iloc[0]["qc_reason"])),
            ssm_dir / f"{model_name}_summary.csv",
        )
        print(f"Skipped {model_name}: trial QC failed")
        return

    p_outlier = p_outlier_setting(config)
    sampling = _sampling_from_args(config, args)
    try:
        _, idata = fit_ssm_model(
            model_trials,
            spec,
            sampling,
            p_outlier=p_outlier,
        )
        summary = summarize_ssm(idata, model_trials, spec, p_outlier=p_outlier)
        write_trace(idata, ssm_dir / f"{model_name}_trace.nc")
        write_csv(summary, ssm_dir / f"{model_name}_summary.csv")
        print(f"Fitted {model_name}: {len(model_trials)} trials")
    except Exception as exc:
        write_csv(
            _empty_summary(model_name, "fit_failed", str(exc)),
            ssm_dir / f"{model_name}_summary.csv",
        )
        raise


def main() -> None:
    args = _parse_args()
    _, output_dir, config = load_runtime_config(args)
    trials_path = output_dir / "app_trials_clean.csv"
    if not trials_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned app trials: {trials_path}. Run app_clean first."
        )
    trials = pd.read_csv(trials_path)
    ssm_dir = output_dir / str(get_ssm_config(config).get("output_subdir", "ssm"))
    ssm_dir.mkdir(parents=True, exist_ok=True)

    model_names = (
        available_model_names(config) if args.model == "all" else [args.model]
    )
    for model_name in model_names:
        _fit_one_model(trials, config, model_name, ssm_dir, args)


if __name__ == "__main__":
    main()
