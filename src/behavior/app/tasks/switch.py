from __future__ import annotations

import re

import pandas as pd

from src.behavior.app.tasks.common import (
    base_metrics,
    correct_rt_stats,
    safe_rate,
    valid_trials,
)


def _rule(task: str, item: object) -> str | None:
    text = str(item)
    if task == "DCCS":
        return text[0] if text and text != "nan" else None
    if task == "DT":
        return "TN" if "T" in text.upper() else "CN"
    if task == "EmotionSwitch":
        match = re.fullmatch(r"LG_(\d+)", text)
        if not match:
            return None
        return "emotion" if int(match.group(1)) <= 4 else "gender"
    return None


def _switch_labels(frame: pd.DataFrame, task: str, mixed_from: int) -> pd.Series:
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    previous_rule: str | None = None
    trial_numbers = pd.to_numeric(frame["trial_index"], errors="coerce")
    for index, item in frame["item"].items():
        if pd.isna(trial_numbers.loc[index]) or trial_numbers.loc[index] < mixed_from:
            continue
        current_rule = _rule(task, item)
        if current_rule is None:
            continue
        if previous_rule is not None:
            labels.loc[index] = "repeat" if current_rule == previous_rule else "switch"
        previous_rule = current_rule
    return labels


def calculate(
    frame: pd.DataFrame, task: str, mixed_from: int
) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    working["switch_condition"] = _switch_labels(working, task, mixed_from)
    metrics = base_metrics(working)
    valid = valid_trials(working)
    for condition, label in (("repeat", "Repeat"), ("switch", "Switch")):
        subset = valid.loc[valid["switch_condition"].eq(condition)]
        rt_mean, _ = correct_rt_stats(subset)
        metrics[f"{label}_ACC"] = safe_rate(subset["correct_trial"])
        metrics[f"{label}_RT"] = rt_mean
    metrics["Switch_Cost_RT"] = metrics["Switch_RT"] - metrics["Repeat_RT"]
    metrics["Switch_Cost_ACC"] = metrics["Switch_ACC"] - metrics["Repeat_ACC"]
    return metrics, []
