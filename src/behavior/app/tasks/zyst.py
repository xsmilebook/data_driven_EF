from __future__ import annotations

import re

import pandas as pd

from src.behavior.app.tasks.common import base_metrics, correct_rt_stats, safe_rate, valid_trials


def _parse_trial_index(value: object) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*([01])\s*", str(value))
    return (int(match.group(1)), int(match.group(2))) if match else None


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    parsed = working["trial_index"].map(_parse_trial_index)
    working["trial"] = parsed.map(lambda value: value[0] if value else pd.NA)
    working["subtrial"] = parsed.map(lambda value: value[1] if value else pd.NA)
    valid = valid_trials(working)

    paired_trials = (
        valid.dropna(subset=["trial", "subtrial"])
        .groupby("trial")["subtrial"]
        .agg(lambda values: set(values.astype(int)))
    )
    paired_ids = paired_trials.loc[paired_trials.map(lambda values: values == {0, 1})].index
    paired = valid.loc[valid["trial"].isin(paired_ids)].copy()
    t0 = paired.loc[paired["subtrial"].eq(0)]
    t1 = paired.loc[paired["subtrial"].eq(1)]
    t0_correct_trials = set(t0.loc[t0["correct_trial"], "trial"])
    t1_given_t0 = t1.loc[t1["trial"].isin(t0_correct_trials)]
    t0_rt, _ = correct_rt_stats(t0)
    t1_rt, _ = correct_rt_stats(t1)

    metrics = base_metrics(valid)
    metrics.update(
        {
            "T0_ACC": safe_rate(t0["correct_trial"]),
            "T1_ACC": safe_rate(t1["correct_trial"]),
            "T1_given_T0_ACC": safe_rate(t1_given_t0["correct_trial"]),
            "T0_RT": t0_rt,
            "T1_RT": t1_rt,
        }
    )
    warnings = [] if len(paired_ids) else ["zyst_complete_pairs_missing"]
    return metrics, warnings
