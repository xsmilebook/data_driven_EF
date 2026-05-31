from __future__ import annotations

import pandas as pd

from src.behavior.app.tasks.common import correct_rt_stats, dprime, safe_rate, valid_trials
from src.common import normalize_bool


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    working["is_go"] = working["answer"].map(normalize_bool)
    valid = valid_trials(working)
    classified = valid.loc[valid["is_go"].notna()].copy()
    go = classified.loc[classified["is_go"].astype(bool)]
    nogo = classified.loc[~classified["is_go"].astype(bool)]
    go_acc = safe_rate(go["correct_trial"])
    nogo_acc = safe_rate(nogo["correct_trial"])
    go_rt_mean, go_rt_sd = correct_rt_stats(go)
    warnings = [] if len(classified) == len(valid) else ["gonogo_unclassified_trials"]
    return (
        {
            "ACC": safe_rate(classified["correct_trial"]),
            "Go_ACC": go_acc,
            "NoGo_ACC": nogo_acc,
            "Go_RT_Mean": go_rt_mean,
            "Go_RT_SD": go_rt_sd,
            "dprime": dprime(go_acc, 1 - nogo_acc),
        },
        warnings,
    )
