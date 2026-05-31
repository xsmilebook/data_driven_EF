from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def valid_trials(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["valid_for_acc"].fillna(False).astype(bool)].copy()


def valid_rt_trials(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["valid_for_rt"].fillna(False).astype(bool)].copy()


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def safe_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else float("nan")


def safe_rate(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if not values.empty else float("nan")


def correct_rt_stats(
    frame: pd.DataFrame, correct_column: str = "correct_trial"
) -> tuple[float, float]:
    rt_valid = valid_rt_trials(frame)
    correct = rt_valid.loc[rt_valid[correct_column].fillna(False).astype(bool), "rt"]
    return safe_mean(correct), safe_sd(correct)


def base_metrics(
    frame: pd.DataFrame, correct_column: str = "correct_trial"
) -> dict[str, float]:
    valid = valid_trials(frame)
    rt_mean, rt_sd = correct_rt_stats(frame, correct_column)
    return {
        "ACC": safe_rate(valid[correct_column]),
        "RT_Mean": rt_mean,
        "RT_SD": rt_sd,
    }


def dprime(hit_rate: float, false_alarm_rate: float) -> float:
    if (
        not np.isfinite(hit_rate)
        or not np.isfinite(false_alarm_rate)
        or hit_rate <= 0
        or hit_rate >= 1
        or false_alarm_rate <= 0
        or false_alarm_rate >= 1
    ):
        return float("nan")
    normal = NormalDist()
    return float(normal.inv_cdf(hit_rate) - normal.inv_cdf(false_alarm_rate))
