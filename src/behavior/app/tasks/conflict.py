from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.behavior.app.tasks.common import (
    base_metrics,
    correct_rt_stats,
    safe_rate,
    valid_trials,
)


def _flanker_condition(item: object) -> str | None:
    match = re.search(r"_([LR])([LR])$", str(item))
    if not match:
        return None
    return "congruent" if match.group(1) == match.group(2) else "incongruent"


def _color_stroop_condition(item: object) -> str | None:
    match = re.fullmatch(r"Pic_([^_]+)_Text_([^_]+)", str(item))
    if not match:
        return None
    return "congruent" if match.group(1) == match.group(2) else "incongruent"


def _condition_series(
    frame: pd.DataFrame, task: str, emotion_mapping: dict[str, Any]
) -> tuple[pd.Series, list[str]]:
    if task == "FLANKER":
        return frame["item"].map(_flanker_condition), []
    if task == "ColorStroop":
        return frame["item"].map(_color_stroop_condition), []
    mapping = {
        str(key): value
        for key, value in emotion_mapping.items()
        if value in {"congruent", "incongruent"}
    }
    warnings = [] if mapping else ["emotion_stroop_condition_mapping_missing"]
    return frame["item"].map(mapping), warnings


def calculate(
    frame: pd.DataFrame, task: str, emotion_mapping: dict[str, Any]
) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    working["condition"], warnings = _condition_series(working, task, emotion_mapping)
    metrics = base_metrics(working)
    valid = valid_trials(working)

    condition_metrics: dict[str, float] = {}
    for condition, label in (("congruent", "Congruent"), ("incongruent", "Incongruent")):
        subset = valid.loc[valid["condition"].eq(condition)]
        rt_mean, _ = correct_rt_stats(subset)
        condition_metrics[f"{label}_ACC"] = safe_rate(subset["correct_trial"])
        condition_metrics[f"{label}_RT"] = rt_mean
    metrics.update(condition_metrics)
    metrics["Contrast_RT"] = metrics["Incongruent_RT"] - metrics["Congruent_RT"]
    metrics["Contrast_ACC"] = metrics["Incongruent_ACC"] - metrics["Congruent_ACC"]
    return metrics, warnings
