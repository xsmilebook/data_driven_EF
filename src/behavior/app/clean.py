from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.behavior.app.check_format import (
    iter_workbooks,
    required_columns,
    subject_code_from_path,
)
from src.behavior.app.task_registry import build_task_registry
from src.common import normalize_text, normalized_equal


OUTPUT_COLUMNS = (
    "dataset",
    "subject_code",
    "workbook",
    "task",
    "trial_index",
    "subject_id",
    "name",
    "item",
    "answer",
    "key",
    "rt",
    "ssrt_or_ssd",
    "blank_screen_duration",
    "total_duration",
    "cross_pic_duration",
    "correct_trial",
    "valid_for_acc",
    "valid_for_rt",
    "exclusion_reason",
)


def _rt_exclusion_reason(value: object, rt_min: float, rt_max: float) -> str:
    if pd.isna(value):
        return "rt_missing_or_non_numeric"
    if value < rt_min:
        return "rt_below_min"
    if value > rt_max:
        return "rt_above_max"
    return ""


def _standardize_sheet(
    dataset: str,
    subject_code: str,
    workbook: Path,
    task: str,
    raw: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = config["columns"]
    reverse_columns = {source: target for target, source in columns.items()}
    frame = raw.rename(columns=reverse_columns).copy()
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["dataset"] = dataset
    frame["subject_code"] = subject_code
    frame["workbook"] = workbook.name
    frame["task"] = task
    for column in ("subject_id", "name", "item", "answer", "key", "ssrt_or_ssd"):
        frame[column] = frame[column].map(normalize_text)

    frame["rt"] = pd.to_numeric(frame["rt"], errors="coerce")
    rt_min = float(config["cleaning"]["rt_min"])
    rt_max = float(config["cleaning"]["rt_max"])
    frame["exclusion_reason"] = frame["rt"].map(
        lambda value: _rt_exclusion_reason(value, rt_min, rt_max)
    )
    valid = frame["exclusion_reason"].eq("")
    frame["valid_for_acc"] = valid
    frame["valid_for_rt"] = valid
    frame["correct_trial"] = [
        normalized_equal(answer, key) for answer, key in zip(frame["answer"], frame["key"])
    ]
    return frame.loc[:, OUTPUT_COLUMNS]


def clean_app_trials(
    dataset: str,
    app_data_dir: Path,
    config: dict[str, Any],
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = build_task_registry(config)
    trial_frames: list[pd.DataFrame] = []
    qc_records: list[dict[str, object]] = []

    for workbook in iter_workbooks(app_data_dir, limit):
        subject_code = subject_code_from_path(workbook)
        excel_file = pd.ExcelFile(workbook)
        for task in excel_file.sheet_names:
            if task not in registry:
                continue
            raw = pd.read_excel(workbook, sheet_name=task)
            missing = sorted(required_columns(task, config["columns"], registry) - set(raw.columns))
            if raw.empty or missing:
                qc_records.append(
                    {
                        "dataset": dataset,
                        "subject_code": subject_code,
                        "task": task,
                        "n_trials_raw": len(raw),
                        "n_trials_rt_valid": 0,
                        "acc_threshold": registry[task].acc_threshold,
                        "ACC": pd.NA,
                        "ok": False,
                        "qc_reason": "empty_sheet" if raw.empty else "missing_columns",
                        "metric_qc_reason": "",
                    }
                )
                continue
            cleaned = _standardize_sheet(dataset, subject_code, workbook, task, raw, config)
            trial_frames.append(cleaned)
            qc_records.append(
                {
                    "dataset": dataset,
                    "subject_code": subject_code,
                    "task": task,
                    "n_trials_raw": len(cleaned),
                    "n_trials_rt_valid": int(cleaned["valid_for_acc"].sum()),
                    "acc_threshold": registry[task].acc_threshold,
                    "ACC": pd.NA,
                    "ok": pd.NA,
                    "qc_reason": "",
                    "metric_qc_reason": "",
                }
            )

    trials = (
        pd.concat(trial_frames, ignore_index=True)
        if trial_frames
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    qc = pd.DataFrame.from_records(qc_records)
    return trials, qc
