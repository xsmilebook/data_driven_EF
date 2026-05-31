from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.behavior.app.task_registry import TaskSpec, build_task_registry


COMMON_REQUIRED_FIELDS = (
    "task",
    "trial_index",
    "subject_id",
    "answer",
    "key",
    "rt",
)


def iter_workbooks(app_data_dir: Path, limit: int | None = None) -> list[Path]:
    workbooks = sorted(
        workbook for workbook in app_data_dir.glob("*.xlsx") if not workbook.name.startswith("~$")
    )
    return workbooks[:limit] if limit is not None else workbooks


def subject_code_from_path(workbook: Path) -> str:
    suffix = "_GameData.xlsx"
    return workbook.name[: -len(suffix)] if workbook.name.endswith(suffix) else workbook.stem


def required_columns(
    task: str, columns: dict[str, str], registry: dict[str, TaskSpec]
) -> set[str]:
    required = {columns[field] for field in COMMON_REQUIRED_FIELDS}
    if task == "SST":
        required.add(columns["ssrt_or_ssd"])
    return required


def inspect_app_format(
    dataset: str,
    app_data_dir: Path,
    config: dict[str, Any],
    limit: int | None = None,
) -> pd.DataFrame:
    registry = build_task_registry(config)
    workbooks = iter_workbooks(app_data_dir, limit)
    subject_counts = Counter(subject_code_from_path(path) for path in workbooks)
    records: list[dict[str, object]] = []

    for workbook in workbooks:
        subject_code = subject_code_from_path(workbook)
        excel_file = pd.ExcelFile(workbook)
        supported_count = 0
        for sheet_name in excel_file.sheet_names:
            supported = sheet_name in registry
            if supported:
                supported_count += 1
            header = pd.read_excel(workbook, sheet_name=sheet_name, nrows=1)
            missing = (
                sorted(required_columns(sheet_name, config["columns"], registry) - set(header.columns))
                if supported
                else []
            )
            reasons: list[str] = []
            if not supported:
                reasons.append("unsupported_task")
            if header.empty:
                reasons.append("empty_sheet")
            if missing:
                reasons.append("missing_columns")
            if subject_counts[subject_code] > 1:
                reasons.append("duplicate_subject_code")
            records.append(
                {
                    "dataset": dataset,
                    "subject_code": subject_code,
                    "workbook": workbook.name,
                    "sheet": sheet_name,
                    "task": sheet_name if supported else pd.NA,
                    "supported": supported,
                    "ok": not reasons,
                    "missing_columns": "|".join(missing),
                    "qc_reason": "|".join(reasons),
                }
            )
        if supported_count == 0:
            records.append(
                {
                    "dataset": dataset,
                    "subject_code": subject_code,
                    "workbook": workbook.name,
                    "sheet": pd.NA,
                    "task": pd.NA,
                    "supported": False,
                    "ok": False,
                    "missing_columns": "",
                    "qc_reason": "no_supported_tasks",
                }
            )
    return pd.DataFrame.from_records(records)
