from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from hssm.param.user_param import UserParam

from src.behavioral_preprocess.ddm.hssm_fit import FitConfig, fit_hssm
from src.behavioral_preprocess.ddm.trials import extract_conflict_multichoice_trials, list_excel_files
from src.behavioral_preprocess.metrics.efny.io import normalize_columns, subject_code_from_filename
from src.behavioral_preprocess.metrics.efny.main import normalize_task_name
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


@dataclass(frozen=True)
class JobSpec:
    task: str
    model_id: str
    model: str
    choices: list[str]
    v_formulas: list[str]
    a_formula: str
    t_formula: str


def _resolve_processed_metrics_root(paths_cfg: dict, roots: dict[str, Path]) -> Path:
    return roots["processed_root"] / "table" / "metrics" / "ssm"


def _fixed_effect_summary(idata: az.InferenceData, *, group_dim: str = "subject") -> pd.DataFrame:
    vars_out: list[str] = []
    for v in idata.posterior.data_vars:
        if "sigma" in v.lower():
            continue
        x = idata.posterior[v]
        if group_dim in x.dims:
            continue
        vars_out.append(v)
    if not vars_out:
        return pd.DataFrame()
    summ = az.summary(idata, var_names=vars_out, hdi_prob=0.95)
    p_gt_0 = {}
    for v in vars_out:
        x = idata.posterior[v].values.reshape(-1)
        p_gt_0[v] = float(np.mean(x > 0))
    summ["p_gt_0"] = pd.Series(p_gt_0)
    summ = summ.reset_index().rename(columns={"index": "var"})
    return summ


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _job_specs() -> list[JobSpec]:
    # Use a 4-choice race model without starting-point bias parameters.
    # This is a practical multi-choice SSM supported by HSSM/SSMs.
    v_re = " + (1|subject)"
    a_re = "a ~ 1 + (1|subject)"
    t_re = "t ~ 1 + (1|subject)"
    return [
        JobSpec(
            task="ColorStroop",
            model_id="colorstroop_race4_null",
            model="race_no_z_4",
            choices=["red", "green", "blue", "yellow"],
            v_formulas=[f"v{i} ~ 1{v_re}" for i in range(4)],
            a_formula=a_re,
            t_formula=t_re,
        ),
        JobSpec(
            task="ColorStroop",
            model_id="colorstroop_race4_congruency_v",
            model="race_no_z_4",
            choices=["red", "green", "blue", "yellow"],
            v_formulas=[f"v{i} ~ 1 + congruency{v_re}" for i in range(4)],
            a_formula=a_re,
            t_formula=t_re,
        ),
        JobSpec(
            task="EmotionStroop",
            model_id="emotionstroop_race4_null",
            model="race_no_z_4",
            choices=["an", "ha", "ne", "sa"],
            v_formulas=[f"v{i} ~ 1{v_re}" for i in range(4)],
            a_formula=a_re,
            t_formula=t_re,
        ),
        JobSpec(
            task="EmotionStroop",
            model_id="emotionstroop_race4_congruency_v",
            model="race_no_z_4",
            choices=["an", "ha", "ne", "sa"],
            v_formulas=[f"v{i} ~ 1 + congruency{v_re}" for i in range(4)],
            a_formula=a_re,
            t_formula=t_re,
        ),
    ]


def _select_job(args: argparse.Namespace) -> JobSpec:
    jobs = _job_specs()
    if args.print_jobs:
        for i, j in enumerate(jobs):
            print(f"[{i}] {j.task} :: {j.model_id}")
        raise SystemExit(0)
    idx = int(args.job_index)
    if idx < 0 or idx >= len(jobs):
        raise ValueError(f"--job-index out of range (0..{len(jobs)-1})")
    return jobs[idx]


def _extract_trials(
    app_dir: Path,
    *,
    task: str,
    choices: list[str],
    pattern: str,
    max_files: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    files = list_excel_files(app_dir, pattern=pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    rows: list[pd.DataFrame] = []
    qc = {"n_files_scanned": 0, "n_files_with_task": 0, "n_trials_kept": 0, "n_subjects": 0}
    for fp in files:
        qc["n_files_scanned"] += 1
        xl = pd.ExcelFile(fp)
        sheet = None
        for sh in xl.sheet_names:
            if normalize_task_name(sh) == task:
                sheet = sh
                break
        if sheet is None:
            continue
        qc["n_files_with_task"] += 1
        subject = subject_code_from_filename(fp.name)
        df = normalize_columns(xl.parse(sheet, dtype="object"))
        d = extract_conflict_multichoice_trials(df, subject=subject, task=task, choices=choices)
        if not d.empty:
            rows.append(d)
    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    qc["n_trials_kept"] = int(len(out))
    qc["n_subjects"] = int(out["subject"].nunique()) if not out.empty else 0
    return out, qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit 4-choice race models for Stroop tasks and save traces/summaries.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--pattern", type=str, default="*.xlsx")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--print-jobs", action="store_true")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job = _select_job(args)

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=None, repo_root=repo_root)
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    app_data_rel = behavioral_cfg.get("app_data_dir")
    if not app_data_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    app_dir = roots["raw_root"] / app_data_rel

    trials, qc = _extract_trials(
        app_dir,
        task=job.task,
        choices=job.choices,
        pattern=args.pattern,
        max_files=args.max_files,
    )
    if trials.empty:
        raise ValueError(f"No trials extracted for task={job.task}.")

    out_root = _resolve_processed_metrics_root(paths_cfg, roots) / args.dataset / job.task / job.model_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"task": job.task, "model_id": job.model_id, "n_trials": len(trials), "qc": qc}, ensure_ascii=False))
        return

    fit_cfg = FitConfig(
        sampler="nuts_numpyro",
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        target_accept=float(args.target_accept),
        random_seed=int(args.seed),
        progress_bar=bool(args.progress_bar),
    )

    include = []
    for vf in job.v_formulas:
        # vf begins with v0/v1/v2/v3
        name = vf.split("~", 1)[0].strip()
        include.append(UserParam(name, formula=vf))
    include.extend(
        [
            UserParam("a", formula=job.a_formula),
            UserParam("t", formula=job.t_formula),
        ]
    )

    res = fit_hssm(trials, model=job.model, include=include, fit_cfg=fit_cfg, loglik_kind="analytical")
    res.idata.to_netcdf(out_root / "idata.nc")
    loo_payload = {}
    if res.loo is not None:
        loo_payload = {
            "elpd_loo": float(res.loo.get("elpd_loo", np.nan)),
            "p_loo": float(res.loo.get("p_loo", np.nan)),
        }
    _save_json(out_root / "loo.json", loo_payload)

    summ = _fixed_effect_summary(res.idata, group_dim="subject")
    if not summ.empty:
        summ.to_csv(out_root / "fixed_effects.csv", index=False)

    run_meta = {
        "dataset": args.dataset,
        "task": job.task,
        "model": job.model,
        "model_id": job.model_id,
        "choices": job.choices,
        "fit": {
            "draws": fit_cfg.draws,
            "tune": fit_cfg.tune,
            "chains": fit_cfg.chains,
            "target_accept": fit_cfg.target_accept,
            "seed": fit_cfg.random_seed,
            "sampler": fit_cfg.sampler,
        },
        "n_trials": int(len(trials)),
        "n_subjects": int(trials["subject"].nunique()),
        "loo": loo_payload,
        "v_formulas": job.v_formulas,
        "a_formula": job.a_formula,
        "t_formula": job.t_formula,
    }
    _save_json(out_root / "run.json", run_meta)
    print(json.dumps(run_meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

