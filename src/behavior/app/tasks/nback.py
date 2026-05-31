from __future__ import annotations

import re

import pandas as pd

from src.behavior.app.tasks.common import base_metrics, dprime, safe_rate, valid_trials


def _stimulus(task: str, item: object) -> object:
    if not task.startswith("Emotion"):
        return item
    match = re.fullmatch(r"Emotion[12]Back_(\d+)", str(item))
    if not match:
        return pd.NA
    return {
        1: "SA",
        2: "NE",
        3: "HA",
        0: "AN",
    }[int(match.group(1)) % 4]


def calculate(
    frame: pd.DataFrame, task: str, n_back: int
) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    working["stimulus"] = working["item"].map(lambda item: _stimulus(task, item))
    previous = working["stimulus"].shift(n_back)
    comparable = working["stimulus"].notna() & previous.notna()
    working["target"] = pd.NA
    working.loc[comparable, "target"] = working.loc[comparable, "stimulus"].eq(
        previous[comparable]
    )

    scored = working.iloc[n_back:].copy()
    metrics = base_metrics(scored)
    valid = valid_trials(scored)
    classified = valid.loc[valid["target"].notna()].copy()
    target = classified.loc[classified["target"].astype(bool)]
    nontarget = classified.loc[~classified["target"].astype(bool)]
    hit_rate = safe_rate(target["correct_trial"])
    false_alarm_rate = safe_rate(~nontarget["correct_trial"].astype(bool))
    metrics.update(
        {
            "Hit_Rate": hit_rate,
            "FA_Rate": false_alarm_rate,
            "dprime": dprime(hit_rate, false_alarm_rate),
        }
    )
    warnings = ["nback_item_missing"] if working["stimulus"].isna().any() else []
    return metrics, warnings
