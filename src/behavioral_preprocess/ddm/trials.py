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


def _norm_str(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


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


def extract_conflict_multichoice_trials(
    df: pd.DataFrame,
    *,
    subject: str,
    task: str,
    choices: list[str],
) -> pd.DataFrame:
    """
    Multi-choice conflict tasks (e.g., Stroop) for race/LBA-type models.

    This keeps RT and the observed response choice (0..n-1), plus congruency.
    """
    if "rt" not in df.columns or "key" not in df.columns or "item" not in df.columns:
        return pd.DataFrame()

    rt = pd.to_numeric(df["rt"], errors="coerce")
    resp = df["key"].map(_norm_str)
    mapping = {c: i for i, c in enumerate([_norm_str(c) for c in choices])}
    response = resp.map(mapping)
    out = pd.DataFrame(
        {
            "subject": subject,
            "task": task,
            "rt": rt,
            "response": response,
        }
    )
    out = out[out["rt"].notna() & out["response"].notna()].copy()
    out["response"] = out["response"].astype(int)

    cond = _conflict_condition(task, df.loc[out.index, "item"])
    out["congruency"] = cond.map({"congruent": 0, "incongruent": 1})
    out = out[out["congruency"].notna()].copy()
    out["congruency"] = out["congruency"].astype(int)
    return out


def extract_sst_go_trials(df: pd.DataFrame, *, subject: str) -> pd.DataFrame:
    """
    Go-only trials for SST.

    The input workbook often stores stop-related quantities in an `SSRT` column. This extractor
    keeps only trials that behave like go trials (SSRT missing / non-numeric), and still requires
    valid RT and a valid correctness-coded response.
    """
    base = _to_ddm_base(df, subject=subject, task="SST")
    if base.empty:
        return base
    if "SSRT" not in df.columns:
        return base
    ssrt = pd.to_numeric(df.loc[base.index, "SSRT"], errors="coerce")
    is_stop = ssrt.notna() & (ssrt > 0)
    out = base.loc[~is_stop].copy()
    out["go_only"] = 1
    return out


def _dt_axis_from_key(key: object) -> str | None:
    k = _norm_str(key)
    if k in {"left", "right"}:
        return "horizontal"
    if k in {"up", "down"}:
        return "vertical"
    return None


def _emotionswitch_dimension_from_key(key: object) -> str | None:
    k = _norm_str(key)
    if k in {"female", "male"}:
        return "gender"
    if k in {"happy", "sad"}:
        return "emotion"
    return None


def _allowed_keys_for_rule(task: str, rule: str) -> set[str]:
    tn = str(task).upper()
    if tn == "DT":
        return {"left", "right"} if rule == "horizontal" else {"down", "up"}
    if tn == "EMOTIONSWITCH":
        return {"female", "male"} if rule == "gender" else {"happy", "sad"}
    return set()


def extract_switch_trials_rule_coded(
    df: pd.DataFrame,
    *,
    subject: str,
    task: str,
    mixed_from: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Extract switch-task trials for DT/EmotionSwitch using rule coding derived from the correct key.

    Outputs a 2-choice dataset suitable for correctness-coded DDM, while excluding responses that
    fall outside the 2-choice sub-task (cross-axis / cross-dimension) as lapse/outliers.

    Returns:
        (trials_df, qc_counts)
    """
    qc = {
        "n_total_rt": 0,
        "n_missing_trial_index": 0,
        "n_missing_key_or_answer": 0,
        "n_missing_rule": 0,
        "n_cross_rule_response": 0,
        "n_kept": 0,
        "n_mixed_kept": 0,
        "n_mixed_with_trial_type": 0,
    }

    base = _to_ddm_base(df, subject=subject, task=task)
    qc["n_total_rt"] = int(len(base))
    if "trial_index" not in df.columns:
        qc["n_missing_trial_index"] = qc["n_total_rt"]
        return base.iloc[0:0].copy(), qc

    trial_idx = pd.to_numeric(df.loc[base.index, "trial_index"], errors="coerce")
    d = df.loc[base.index].copy()
    d["_trial_index_num"] = trial_idx
    # normalize_columns convention:
    # - answer: correct answer (ground truth)
    # - key: subject keypress
    d["_correct_norm"] = d.get("answer", pd.Series(pd.NA, index=d.index)).map(_norm_str)
    d["_resp_norm"] = d.get("key", pd.Series(pd.NA, index=d.index)).map(_norm_str)

    keep = d["_trial_index_num"].notna()
    qc["n_missing_trial_index"] = int((~keep).sum())
    d = d[keep].copy()

    keep = (d["_correct_norm"] != "") & (d["_resp_norm"] != "")
    qc["n_missing_key_or_answer"] = int((~keep).sum())
    d = d[keep].copy()

    tn = str(task).upper()
    if tn == "DT":
        d["rule"] = d["answer"].map(_dt_axis_from_key)
    elif tn == "EMOTIONSWITCH":
        d["rule"] = d["answer"].map(_emotionswitch_dimension_from_key)
    else:
        d["rule"] = None

    keep = d["rule"].notna()
    qc["n_missing_rule"] = int((~keep).sum())
    d = d[keep].copy()

    ans = d["_resp_norm"]
    rule = d["rule"].astype(str)
    if tn == "DT":
        in_rule = (rule.eq("horizontal") & ans.isin(["left", "right"])) | (rule.eq("vertical") & ans.isin(["up", "down"]))
    elif tn == "EMOTIONSWITCH":
        in_rule = (rule.eq("gender") & ans.isin(["female", "male"])) | (rule.eq("emotion") & ans.isin(["happy", "sad"]))
    else:
        in_rule = pd.Series(False, index=d.index, dtype="boolean")
    in_rule = in_rule.astype("boolean")
    qc["n_cross_rule_response"] = int((~in_rule).sum())
    d = d[in_rule.fillna(False)].copy()

    out = base.loc[d.index].copy()
    out["trial_index"] = d["_trial_index_num"].astype(int).values
    out["block"] = np.where(out["trial_index"] >= int(mixed_from), "mixed", "pure")
    out["rule"] = d["rule"].astype(str).values

    # repeat/switch defined by rule transitions within mixed block only (drop first mixed trial)
    d_sorted = out.sort_values("trial_index").copy()
    prev_rule = d_sorted["rule"].shift(1)
    prev_block = d_sorted["block"].shift(1)
    trial_type = pd.Series(pd.NA, index=d_sorted.index, dtype="object")
    in_mixed = d_sorted["block"].eq("mixed") & prev_block.eq("mixed") & prev_rule.notna()
    trial_type.loc[in_mixed & d_sorted["rule"].eq(prev_rule)] = "repeat"
    trial_type.loc[in_mixed & ~d_sorted["rule"].eq(prev_rule)] = "switch"
    d_sorted["trial_type"] = trial_type
    out = d_sorted.sort_index()

    qc["n_kept"] = int(len(out))
    qc["n_mixed_kept"] = int((out["block"] == "mixed").sum())
    qc["n_mixed_with_trial_type"] = int(out["trial_type"].notna().sum())
    return out, qc


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
        "switch_rule_coded": [],
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
            out["switch_mixing"].append(extract_switch_trials_mixing(df, subject=subject, task=task, mixed_from=mixed_from))
            out["switch_only"].append(extract_switch_trials_switch_only(df, subject=subject, task=task, mixed_from=mixed_from))
            if task in {"DT", "EmotionSwitch"}:
                rule_coded, _ = extract_switch_trials_rule_coded(
                    df, subject=subject, task=task, mixed_from=mixed_from
                )
                out["switch_rule_coded"].append(rule_coded)

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
    switch_rule_coded = []
    for fp in files:
        res = extract_trials_from_workbook(fp, task_config=task_config, cfg=cfg)
        if not res["conflict"].empty:
            conflict.append(res["conflict"])
        if not res["switch_mixing"].empty:
            switch_mixing.append(res["switch_mixing"])
        if not res["switch_only"].empty:
            switch_only.append(res["switch_only"])
        if "switch_rule_coded" in res and not res["switch_rule_coded"].empty:
            switch_rule_coded.append(res["switch_rule_coded"])

    return {
        "conflict": pd.concat(conflict, axis=0, ignore_index=True) if conflict else pd.DataFrame(),
        "switch_mixing": pd.concat(switch_mixing, axis=0, ignore_index=True) if switch_mixing else pd.DataFrame(),
        "switch_only": pd.concat(switch_only, axis=0, ignore_index=True) if switch_only else pd.DataFrame(),
        "switch_rule_coded": (
            pd.concat(switch_rule_coded, axis=0, ignore_index=True) if switch_rule_coded else pd.DataFrame()
        ),
    }
