from __future__ import annotations

import pandas as pd

from src.behavior.app.tasks.common import base_metrics, correct_rt_stats, safe_rate, valid_trials
from src.common import normalize_text


def _lower(value: object) -> str | None:
    text = normalize_text(value)
    return text.casefold() if text else None


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    valid = valid_trials(frame)
    answer = valid["answer"].map(_lower)
    key = valid["key"].map(_lower)
    answer_right = valid.loc[answer.eq("right")]
    answer_left = valid.loc[answer.eq("left")]
    metrics = base_metrics(valid)
    metrics.update(
        {
            "Miss_Rate": safe_rate(~answer_right["key"].map(_lower).eq("right")),
            "FA_Rate": safe_rate(answer_left["key"].map(_lower).eq("right")),
        }
    )
    correct_rt_mean, correct_rt_sd = correct_rt_stats(valid)
    metrics["Correct_RT_Mean"] = correct_rt_mean
    metrics["Correct_RT_SD"] = correct_rt_sd
    return metrics, []
