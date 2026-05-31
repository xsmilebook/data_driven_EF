from __future__ import annotations

import numpy as np
import pandas as pd

from src.behavior.app.tasks.common import correct_rt_stats, safe_mean, safe_rate, valid_trials
from src.common import normalize_text, normalized_equal


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    working = frame.copy()
    working["ssd_numeric"] = pd.to_numeric(working["ssrt_or_ssd"], errors="coerce")
    working["is_stop"] = working["ssd_numeric"].notna()
    working["sst_correct"] = [
        normalize_text(key) is None if is_stop else normalized_equal(answer, key)
        for is_stop, answer, key in zip(working["is_stop"], working["answer"], working["key"])
    ]

    valid = valid_trials(working)
    stop = valid.loc[valid["is_stop"]]
    go = valid.loc[~valid["is_stop"]]
    correct_go = go.loc[go["sst_correct"] & go["valid_for_rt"]]
    go_rt_mean, go_rt_sd = correct_rt_stats(go, "sst_correct")
    stop_acc = safe_rate(stop["sst_correct"])
    mean_ssd = safe_mean(stop["ssd_numeric"])

    ssrt = float("nan")
    warnings: list[str] = []
    if stop.empty:
        warnings.append("sst_stop_trials_missing")
    if correct_go.empty:
        warnings.append("sst_correct_go_trials_missing")
    if not stop.empty and not correct_go.empty:
        probability = 1 - stop_acc
        ssrt = float(np.quantile(correct_go["rt"], probability) - mean_ssd)

    return (
        {
            "ACC": safe_rate(valid["sst_correct"]),
            "Stop_ACC": stop_acc,
            "Mean_SSD": mean_ssd,
            "SSRT": ssrt,
            "Go_RT_Mean": go_rt_mean,
            "Go_RT_SD": go_rt_sd,
        },
        warnings,
    )
