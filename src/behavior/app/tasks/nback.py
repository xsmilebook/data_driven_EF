from __future__ import annotations

import pandas as pd

from src.behavior.app.tasks.common import base_metrics, dprime, safe_rate, valid_trials


def calculate(frame: pd.DataFrame, n_back: int) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    previous = working["item"].shift(n_back)
    comparable = working["item"].notna() & previous.notna()
    working["target"] = pd.NA
    working.loc[comparable, "target"] = working.loc[comparable, "item"].eq(previous[comparable])

    metrics = base_metrics(working)
    valid = valid_trials(working)
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
    return metrics, []
