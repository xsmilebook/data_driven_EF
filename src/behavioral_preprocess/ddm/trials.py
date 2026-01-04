from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavioral_preprocess.metrics.efny.io import normalize_columns, subject_code_from_filename
from src.behavioral_preprocess.metrics.efny.metrics import _conflict_condition
from src.behavioral_preprocess.metrics.efny.main import normalize_task_name


@dataclass(frozen=True)
class DDMExtractionConfig:
    mixed_from_by_task: dict[str, int]


CONFLICT_TASKS = ("FLANKER", "ColorStroop", "EmotionStroop")
SWITCH_TASKS = ("DT", "EmotionSwitch", "DCCS")


def list_excel_files(app_data_dir: Path, pattern: str = "*.xlsx") -> list[Path]:
    files = sorted(app_data_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def _normalize_bool_correct(df: pd.DataFrame) -> pd.Series:
    if "answer" not in df.columns or "key" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    ans = df["answer"].astype(str).str.strip().str.lower()
    key = df["key"].astype(str).str.strip().str.lower()
    correct = ans.eq(key)
    return pd.Series(correct, index=df.index, dtype="boolean")


def _to_ddm_base(df: pd.DataFrame, subject: str, task: str) -> pd.DataFrame:
    d = df.copy()
    d["rt"] = pd.to_numeric(d.get("rt", pd.Series(pd.NA, index=d.index)), errors="coerce")
    correct = _normalize_bool_correct(d)
    response = pd.Series(np.nan, index=d.index, dtype="float")
    response.loc[correct == True] = 1
    response.loc[correct == False] = -1

    out = pd.DataFrame(
        {
            "subject": subject,
            "task": task,
            "rt": d["rt"],
            "response": response,
        }
    )
    out = out[out["rt"].notna() & out["response"].notna()].copy()
    out["response"] = out["response"].astype(int)
    return out


def extract_conflict_trials(df: pd.DataFrame, *, subject: str, task: str) -> pd.DataFrame:
    base = _to_ddm_base(df, subject=subject, task=task)
    if "item" not in df.columns:
        return base.iloc[0:0].copy()
    cond = _conflict_condition(task, df.loc[base.index, "item"])
    base = base.copy()
    base["congruency"] = cond.map({"congruent": 0, "incongruent": 1})
    base = base[base["congruency"].notna()].copy()
    base["congruency"] = base["congruency"].astype(int)
    return base


def _switch_rule_series(d: pd.DataFrame, task: str) -> pd.Series:
    tn = str(task)
    rule = pd.Series(np.nan, index=d.index, dtype="object")
    if tn.upper() == "DCCS":
        rule = d["item"].astype(str).str.slice(0, 1)
    elif tn.upper() == "DT":
        rule = np.where(d["item"].astype(str).str.contains("[Tt]"), "TN", "CN")
        rule = pd.Series(rule, index=d.index, dtype="object")
    elif tn.upper() == "EMOTIONSWITCH":
        num = d["item"].astype(str).str.extract(r"(\d+)")[0]
        num = pd.to_numeric(num, errors="coerce")
        rule = pd.Series(np.nan, index=d.index, dtype="object")
        em = num.notna() & num.between(1, 4)
        ge = num.notna() & num.between(5, 8)
        rule.loc[em] = "emotion"
        rule.loc[ge] = "gender"
    return rule


def extract_switch_trials_mixing(
    df: pd.DataFrame,
    *,
    subject: str,
    task: str,
    mixed_from: int,
) -> pd.DataFrame:
    base = _to_ddm_base(df, subject=subject, task=task)
    if "trial_index" not in df.columns:
        return base.iloc[0:0].copy()
    trial_idx = pd.to_numeric(df.loc[base.index, "trial_index"], errors="coerce")
    base = base.copy()
    base["trial_index"] = trial_idx
    base = base[base["trial_index"].notna()].copy()
    base["trial_index"] = base["trial_index"].astype(int)
    base["block_mixed"] = (base["trial_index"] >= int(mixed_from)).astype(int)
    return base


def extract_switch_trials_switch_only(
    df: pd.DataFrame,
    *,
    subject: str,
    task: str,
    mixed_from: int,
) -> pd.DataFrame:
    base = _to_ddm_base(df, subject=subject, task=task)
    if "trial_index" not in df.columns or "item" not in df.columns:
        return base.iloc[0:0].copy()

    d = df.loc[base.index].copy()
    d["_trial_index_num"] = pd.to_numeric(d["trial_index"], errors="coerce")
    d = d[d["_trial_index_num"].notna()].copy()
    d = d.sort_values("_trial_index_num")

    rule = _switch_rule_series(d, task=task)
    prev_rule = pd.Series(rule).shift(1)
    switch_type = pd.Series(np.nan, index=d.index, dtype="object")
    same = (rule == prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    diff = (rule != prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    switch_type.loc[same] = "repeat"
    switch_type.loc[diff] = "switch"

    d["switch_type"] = switch_type
    d = d[d["switch_type"].notna()].copy()
    d = d[d["_trial_index_num"] >= float(mixed_from)].copy()
    if len(d) == 0:
        return base.iloc[0:0].copy()

    out = base.loc[d.index].copy()
    out["trial_index"] = d["_trial_index_num"].astype(int).values
    out["is_switch"] = (d["switch_type"] == "switch").astype(int).values
    return out


def extract_trials_from_workbook(
    excel_path: Path,
    *,
    task_config: dict[str, dict],
    cfg: DDMExtractionConfig,
) -> dict[str, pd.DataFrame]:
    subject = subject_code_from_filename(excel_path.name)
    xl = pd.ExcelFile(excel_path)
    out: dict[str, list[pd.DataFrame]] = {
        "conflict": [],
        "switch_mixing": [],
        "switch_only": [],
    }

    for sheet in xl.sheet_names:
        task = normalize_task_name(sheet)
        if task not in task_config:
            continue
        t = task_config[task].get("type")
        if t not in {"conflict", "switch"}:
            continue
        df_raw = xl.parse(sheet, dtype="object")
        df = normalize_columns(df_raw)
        if t == "conflict":
            out["conflict"].append(extract_conflict_trials(df, subject=subject, task=task))
        elif t == "switch":
            mixed_from = cfg.mixed_from_by_task.get(task)
            if mixed_from is None:
                continue
            out["switch_mixing"].append(
                extract_switch_trials_mixing(df, subject=subject, task=task, mixed_from=mixed_from)
            )
            out["switch_only"].append(
                extract_switch_trials_switch_only(df, subject=subject, task=task, mixed_from=mixed_from)
            )

    return {k: pd.concat(v, axis=0, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}


def extract_trials_from_directory(
    app_data_dir: Path,
    *,
    task_config: dict[str, dict],
    cfg: DDMExtractionConfig,
    pattern: str = "*.xlsx",
    max_files: int | None = None,
) -> dict[str, pd.DataFrame]:
    files = list_excel_files(app_data_dir, pattern=pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    conflict = []
    switch_mixing = []
    switch_only = []
    for fp in files:
        res = extract_trials_from_workbook(fp, task_config=task_config, cfg=cfg)
        if not res["conflict"].empty:
            conflict.append(res["conflict"])
        if not res["switch_mixing"].empty:
            switch_mixing.append(res["switch_mixing"])
        if not res["switch_only"].empty:
            switch_only.append(res["switch_only"])

    return {
        "conflict": pd.concat(conflict, axis=0, ignore_index=True) if conflict else pd.DataFrame(),
        "switch_mixing": pd.concat(switch_mixing, axis=0, ignore_index=True) if switch_mixing else pd.DataFrame(),
        "switch_only": pd.concat(switch_only, axis=0, ignore_index=True) if switch_only else pd.DataFrame(),
    }
