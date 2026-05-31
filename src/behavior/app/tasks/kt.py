from __future__ import annotations

import pandas as pd

from src.behavior.app.tasks.common import base_metrics


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    metrics = base_metrics(frame)
    metrics["Overall_ACC"] = metrics["ACC"]
    metrics["Mean_RT"] = metrics["RT_Mean"]
    return metrics, []
