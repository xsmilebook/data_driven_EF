from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.behavioral_preprocess.ddm.hssm_fit import FitConfig, fit_hssm
from src.behavioral_preprocess.ddm.trials import (
    extract_conflict_trials,
    extract_sst_go_trials,
    extract_switch_trials_rule_coded,
    list_excel_files,
)
from src.behavioral_preprocess.metrics.efny.io import normalize_columns, subject_code_from_filename
from src.behavioral_preprocess.metrics.efny.main import load_task_config, normalize_task_name
from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


@dataclass(frozen=True)
class JobSpec:
    task: str
    dataset_kind: str
    model: str
    model_id: str
    v_formula: str
    a_formula: str
    t_formula: str
    z_formula: str


def _resolve_processed_metrics_root(paths_cfg: dict, roots: dict[str, Path]) -> Path:
    return roots["processed_root"] / "table" / "metrics" / "ssm"


def _coerce_categories(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def _extract_single_task_trials(
    app_dir: Path,
    *,
    task: str,
    task_cfg: dict[str, dict],
    pattern: str,
    max_files: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    files = list_excel_files(app_dir, pattern=pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    mixed_from = None
    if task in task_cfg and "mixed_from" in task_cfg[task]:
        mixed_from = int(task_cfg[task]["mixed_from"])

    rows: list[pd.DataFrame] = []
    qc_total: dict[str, int] = {
        "n_files_scanned": 0,
        "n_files_with_task": 0,
        "n_trials_kept": 0,
        "n_subjects": 0,
    }
    qc_extra: dict[str, int] = {}

    for fp in files:
        qc_total["n_files_scanned"] += 1
        xl = pd.ExcelFile(fp)
        sheet = None
        for sh in xl.sheet_names:
            if normalize_task_name(sh) == task:
                sheet = sh
                break
        if sheet is None:
            continue

        qc_total["n_files_with_task"] += 1
        subject = subject_code_from_filename(fp.name)
        df_raw = xl.parse(sheet, dtype="object")
        df = normalize_columns(df_raw)

        if task == "SST":
            d = extract_sst_go_trials(df, subject=subject)
            if not d.empty:
                rows.append(d)
            continue

        if task in {"DT", "EmotionSwitch"}:
            if mixed_from is None:
                raise ValueError(f"Missing mixed_from for {task}")
            d, qc = extract_switch_trials_rule_coded(df, subject=subject, task=task, mixed_from=mixed_from)
            qc_extra = {k: qc_extra.get(k, 0) + int(v) for k, v in qc.items()}
            if not d.empty:
                rows.append(d)
            continue

        if task == "FLANKER":
            d = extract_conflict_trials(df, subject=subject, task=task)
            if not d.empty:
                rows.append(d)
            continue

    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    qc_total["n_trials_kept"] = int(len(out))
    qc_total["n_subjects"] = int(out["subject"].nunique()) if not out.empty and "subject" in out.columns else 0
    qc_total.update(qc_extra)
    return out, qc_total


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
    # NOTE: These specs implement the current DDM design in docs/reports/ddm_decision.md.
    # DDM parameter names use HSSM conventions: non-decision time is `t`.
    z_re = "z ~ 1 + (1|subject)"
    a_re = "a ~ 1 + (1|subject)"
    t_re = "t ~ 1 + (1|subject)"
    return [
        # FLANKER: null vs congruency effect on v (a/t fixed across congruency)
        JobSpec(
            task="FLANKER",
            dataset_kind="conflict_all",
            model="ddm",
            model_id="flanker_null",
            v_formula="v ~ 1 + (1|subject)",
            a_formula=a_re,
            t_formula=t_re,
            z_formula=z_re,
        ),
        JobSpec(
            task="FLANKER",
            dataset_kind="conflict_all",
            model="ddm",
            model_id="flanker_congruency_v",
            v_formula="v ~ 1 + congruency + (1|subject)",
            a_formula=a_re,
            t_formula=t_re,
            z_formula=z_re,
        ),
        # SST: go-only DDM (single model; compare elsewhere if needed)
        JobSpec(
            task="SST",
            dataset_kind="sst_go_only",
            model="ddm",
            model_id="sst_go_only",
            v_formula="v ~ 1 + (1|subject)",
            a_formula=a_re,
            t_formula=t_re,
            z_formula=z_re,
        ),
        # DT: Model A (Mixing) null vs effect; v/a/t all vary with condition and rule
        JobSpec(
            task="DT",
            dataset_kind="switch_rule_all",
            model="ddm",
            model_id="dt_mixing_null",
            v_formula="v ~ 1 + rule + (1|subject)",
            a_formula="a ~ 1 + rule + (1|subject)",
            t_formula="t ~ 1 + rule + (1|subject)",
            z_formula=z_re,
        ),
        JobSpec(
            task="DT",
            dataset_kind="switch_rule_all",
            model="ddm",
            model_id="dt_mixing",
            v_formula="v ~ 1 + block + rule + block:rule + (1|subject)",
            a_formula="a ~ 1 + block + rule + block:rule + (1|subject)",
            t_formula="t ~ 1 + block + rule + block:rule + (1|subject)",
            z_formula=z_re,
        ),
        # DT: Model B (Switch) null vs effect; mixed only
        JobSpec(
            task="DT",
            dataset_kind="switch_rule_mixed",
            model="ddm",
            model_id="dt_switch_null",
            v_formula="v ~ 1 + rule + (1|subject)",
            a_formula="a ~ 1 + rule + (1|subject)",
            t_formula="t ~ 1 + rule + (1|subject)",
            z_formula=z_re,
        ),
        JobSpec(
            task="DT",
            dataset_kind="switch_rule_mixed",
            model="ddm",
            model_id="dt_switch",
            v_formula="v ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            a_formula="a ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            t_formula="t ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            z_formula=z_re,
        ),
        # EmotionSwitch: Model A (Mixing)
        JobSpec(
            task="EmotionSwitch",
            dataset_kind="switch_rule_all",
            model="ddm",
            model_id="emotionswitch_mixing_null",
            v_formula="v ~ 1 + rule + (1|subject)",
            a_formula="a ~ 1 + rule + (1|subject)",
            t_formula="t ~ 1 + rule + (1|subject)",
            z_formula=z_re,
        ),
        JobSpec(
            task="EmotionSwitch",
            dataset_kind="switch_rule_all",
            model="ddm",
            model_id="emotionswitch_mixing",
            v_formula="v ~ 1 + block + rule + block:rule + (1|subject)",
            a_formula="a ~ 1 + block + rule + block:rule + (1|subject)",
            t_formula="t ~ 1 + block + rule + block:rule + (1|subject)",
            z_formula=z_re,
        ),
        # EmotionSwitch: Model B (Switch)
        JobSpec(
            task="EmotionSwitch",
            dataset_kind="switch_rule_mixed",
            model="ddm",
            model_id="emotionswitch_switch_null",
            v_formula="v ~ 1 + rule + (1|subject)",
            a_formula="a ~ 1 + rule + (1|subject)",
            t_formula="t ~ 1 + rule + (1|subject)",
            z_formula=z_re,
        ),
        JobSpec(
            task="EmotionSwitch",
            dataset_kind="switch_rule_mixed",
            model="ddm",
            model_id="emotionswitch_switch",
            v_formula="v ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            a_formula="a ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            t_formula="t ~ 1 + trial_type + rule + trial_type:rule + (1|subject)",
            z_formula=z_re,
        ),
    ]


def _select_job(args: argparse.Namespace) -> JobSpec:
    jobs = _job_specs()
    if args.print_jobs:
        for i, j in enumerate(jobs):
            print(f"[{i}] {j.task} :: {j.model_id} ({j.dataset_kind})")
        raise SystemExit(0)
    if args.job_index is not None:
        idx = int(args.job_index)
        if idx < 0 or idx >= len(jobs):
            raise ValueError(f"--job-index out of range (0..{len(jobs)-1})")
        return jobs[idx]
    if args.task and args.model_id:
        for j in jobs:
            if j.task == args.task and j.model_id == args.model_id:
                return j
        raise ValueError("No matching job for --task/--model-id; use --print-jobs.")
    raise ValueError("Specify --job-index or (--task and --model-id).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit task-level hierarchical SSM models and save traces/summaries.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--metrics-config", dest="metrics_config", type=str, default="configs/behavioral_metrics.yaml")
    parser.add_argument("--pattern", type=str, default="*.xlsx")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--print-jobs", action="store_true")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Extract and report counts without sampling/writing traces.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=None, repo_root=repo_root)
    behavioral_cfg = dataset_cfg.get("behavioral", {})
    app_data_rel = behavioral_cfg.get("app_data_dir")
    if not app_data_rel:
        raise ValueError("Missing dataset.behavioral.app_data_dir in configs/paths.yaml.")
    app_dir = roots["raw_root"] / app_data_rel

    task_cfg = load_task_config(args.metrics_config)
    job = _select_job(args)

    trials, qc = _extract_single_task_trials(
        app_dir,
        task=job.task,
        task_cfg=task_cfg,
        pattern=args.pattern,
        max_files=args.max_files,
    )
    if trials.empty:
        raise ValueError(f"No trials extracted for task={job.task}.")

    if job.task == "FLANKER":
        trials["congruency"] = trials["congruency"].astype(int)
    if job.task in {"DT", "EmotionSwitch"}:
        if job.dataset_kind == "switch_rule_mixed":
            trials = trials[(trials["block"] == "mixed") & trials["trial_type"].notna()].copy()
        trials = _coerce_categories(trials, ["block", "rule", "trial_type"])

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

    include = [
        # Core DDM parameters (t == t0)
        # HSSM will parse the formula and build the appropriate transforms internally.
        # Keep z free (with random effects) per project decision.
        # noqa: E501
    ]
    from hssm.param.user_param import UserParam  # local import to keep script import-time light

    include = [
        UserParam("v", formula=job.v_formula),
        UserParam("a", formula=job.a_formula),
        UserParam("t", formula=job.t_formula),
        UserParam("z", formula=job.z_formula),
    ]

    res = fit_hssm(trials, model=job.model, include=include, fit_cfg=fit_cfg, loglik_kind="analytical")

    idata_path = out_root / "idata.nc"
    res.idata.to_netcdf(idata_path)

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
        "dataset_kind": job.dataset_kind,
        "v_formula": job.v_formula,
        "a_formula": job.a_formula,
        "t_formula": job.t_formula,
        "z_formula": job.z_formula,
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
    }
    _save_json(out_root / "run.json", run_meta)
    print(json.dumps(run_meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

