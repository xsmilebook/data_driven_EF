from __future__ import annotations

import pandas as pd

from src.behavior.app.tasks.common import safe_rate, valid_trials


def calculate(frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    valid = valid_trials(frame)
    return {"ACC": safe_rate(valid["correct_trial"])}, []
