from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.behavior.app.task_registry import TaskSpec, build_task_registry
from src.behavior.app.tasks import conflict, fzss, gonogo, kt, nback, sst, switch, zyst


def _calculate_task_metrics(
    frame: pd.DataFrame, spec: TaskSpec, config: dict[str, Any]
) -> tuple[dict[str, float], list[str]]:
    if spec.family == "nback":
        return nback.calculate(frame, spec.n_back)
    if spec.family == "conflict":
        return conflict.calculate(frame, spec.name, config["emotion_stroop_conditions"])
    if spec.family == "switch":
        return switch.calculate(frame, spec.name, spec.mixed_from)
    if spec.family == "sst":
        return sst.calculate(frame)
    if spec.family == "gonogo":
        return gonogo.calculate(frame)
    if spec.family == "zyst":
        return zyst.calculate(frame)
    if spec.family == "fzss":
        return fzss.calculate(frame)
    if spec.family == "kt":
        return kt.calculate(frame)
    raise ValueError(f"Unsupported task family: {spec.family}")


def _metric_qc_reason(warnings: list[str]) -> str:
    return "|".join(sorted(set(warnings)))


def calculate_app_metrics(
    trials: pd.DataFrame,
    clean_qc: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = build_task_registry(config)
    metric_records: list[dict[str, object]] = []
    metric_qc_records: list[dict[str, object]] = []

    for (dataset, subject_code, task), frame in trials.groupby(
        ["dataset", "subject_code", "task"], sort=True
    ):
        spec = registry[task]
        metrics, warnings = _calculate_task_metrics(frame, spec, config)
        acc = metrics["ACC"]
        if not np.isfinite(acc):
            ok = False
            warnings.append("acc_missing")
        elif spec.acc_threshold is not None and acc < spec.acc_threshold:
            ok = False
            warnings.append("acc_below_random_threshold")
        else:
            ok = True

        metric_qc_records.append(
            {
                "dataset": dataset,
                "subject_code": subject_code,
                "task": task,
                "n_trials_raw": len(frame),
                "n_trials_rt_valid": int(frame["valid_for_acc"].fillna(False).astype(bool).sum()),
                "acc_threshold": spec.acc_threshold,
                "ACC": acc,
                "ok": ok,
                "qc_reason": "" if ok else "task_metrics_invalid",
                "metric_qc_reason": _metric_qc_reason(warnings),
            }
        )
        for metric, value in metrics.items():
            metric_records.append(
                {
                    "dataset": dataset,
                    "subject_code": subject_code,
                    "task": task,
                    "metric": metric,
                    "value": value if ok else float("nan"),
                }
            )

    metrics_long = pd.DataFrame.from_records(
        metric_records, columns=["dataset", "subject_code", "task", "metric", "value"]
    )
    if metrics_long.empty:
        metrics_wide = pd.DataFrame(columns=["dataset", "subject_code"])
    else:
        wide_source = metrics_long.assign(
            task_metric=metrics_long["task"] + "_" + metrics_long["metric"]
        )
        metrics_wide = (
            wide_source.pivot(
                index=["dataset", "subject_code"], columns="task_metric", values="value"
            )
            .reset_index()
            .rename_axis(columns=None)
        )

    calculated_qc = pd.DataFrame.from_records(metric_qc_records)
    if clean_qc.empty:
        task_qc = calculated_qc
    else:
        keys = ["dataset", "subject_code", "task"]
        untouched = clean_qc.merge(calculated_qc[keys], on=keys, how="left", indicator=True)
        untouched = untouched.loc[untouched["_merge"].eq("left_only"), clean_qc.columns]
        task_qc = pd.concat([calculated_qc, untouched], ignore_index=True)
    return metrics_long, metrics_wide, task_qc
