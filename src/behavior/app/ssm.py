from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from src.behavior.app.task_registry import build_task_registry
from src.behavior.app.tasks import conflict
from src.common import normalize_bool, normalize_text, normalized_equal


DDM_RESPONSE = {False: -1, True: 1}
NBACK_DOMAIN_TASKS = {
    "number": {"Number1Back", "Number2Back"},
    "spatial": {"Spatial1Back", "Spatial2Back"},
    "emotion": {"Emotion1Back", "Emotion2Back"},
}


@dataclass(frozen=True)
class SSMModelSpec:
    name: str
    family: str
    hssm_model: str
    formulas: dict[str, str]
    task: str | None = None
    domain: str | None = None
    choices: dict[str, int] | None = None


def get_ssm_config(config: dict[str, Any]) -> dict[str, Any]:
    ssm_config = config.get("ssm_models")
    if not isinstance(ssm_config, dict):
        raise ValueError("Missing `ssm_models` section in metrics config.")
    return ssm_config


def get_model_spec(config: dict[str, Any], model_name: str) -> SSMModelSpec:
    models = get_ssm_config(config).get("models", {})
    if model_name not in models:
        raise ValueError(f"Unknown SSM model: {model_name}")
    values = models[model_name]
    return SSMModelSpec(
        name=model_name,
        family=values["family"],
        hssm_model=values["hssm_model"],
        formulas=dict(values.get("formulas") or {}),
        task=values.get("task"),
        domain=values.get("domain"),
        choices=dict(values.get("choices") or {}) or None,
    )


def available_model_names(config: dict[str, Any]) -> list[str]:
    return sorted(get_ssm_config(config).get("models", {}).keys())


def sampling_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    values = get_ssm_config(config).get("sampling", {}).get(mode)
    if not isinstance(values, dict):
        raise ValueError(f"Unknown SSM sampling mode: {mode}")
    return dict(values)


def _base_rt_valid(trials: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rt_min = float(config["cleaning"]["rt_min"])
    rt_max = float(config["cleaning"]["rt_max"])
    working = trials.copy()
    working["rt"] = pd.to_numeric(working["rt"], errors="coerce")
    return working.loc[
        working["valid_for_rt"].fillna(False).astype(bool)
        & working["rt"].between(rt_min, rt_max, inclusive="both")
    ].copy()


def _as_ddm_trials(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["choice"] = [
        normalized_equal(answer, key)
        for answer, key in zip(working["answer"], working["key"])
    ]
    working["response"] = working["choice"].map(DDM_RESPONSE)
    return working


def _trial_numbers(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["trial_index"], errors="coerce")


def _switch_trial_type(frame: pd.DataFrame, mixed_from: int) -> pd.Series:
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    previous_rule: str | None = None
    trial_numbers = _trial_numbers(frame)
    for index, rule in frame["rule"].items():
        if pd.isna(trial_numbers.loc[index]) or trial_numbers.loc[index] < mixed_from:
            continue
        if pd.isna(rule):
            continue
        if previous_rule is not None:
            labels.loc[index] = "repeat" if rule == previous_rule else "switch"
        previous_rule = str(rule)
    return labels


def _switch_rule(task: str, correct_answer: object) -> str | None:
    answer = normalize_text(correct_answer)
    if answer is None:
        return None
    normalized = answer.casefold()
    if task == "DT":
        if normalized in {"left", "right"}:
            return "horizontal"
        if normalized in {"up", "down"}:
            return "vertical"
    if task == "EmotionSwitch":
        if normalized in {"female", "male"}:
            return "gender"
        if normalized in {"happy", "sad"}:
            return "emotion"
    return None


def _nback_load(task: str) -> int | None:
    match = re.search(r"([12])Back$", task)
    return int(match.group(1)) if match else None


def _time_bins(frame: pd.DataFrame, bins: int = 4) -> pd.Series:
    trial_numbers = _trial_numbers(frame)
    ranked = trial_numbers.rank(method="first")
    unique_count = ranked.nunique(dropna=True)
    if unique_count < 2:
        return pd.Series("bin1", index=frame.index, dtype="object")
    n_bins = min(bins, int(unique_count))
    labels = [f"bin{i + 1}" for i in range(n_bins)]
    return pd.qcut(ranked, q=n_bins, labels=labels, duplicates="drop").astype("object")


def _prepare_flanker(frame: pd.DataFrame) -> pd.DataFrame:
    working = _as_ddm_trials(frame)
    working["congruency"] = working["item"].map(conflict._flanker_condition)
    return working.loc[working["congruency"].notna()].copy()


def _prepare_dccs(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    working = _as_ddm_trials(frame)
    if family == "dccs_block":
        trial_numbers = _trial_numbers(working)
        working["block"] = np.where(trial_numbers <= 20, "pure", "mixed")
        working = working.loc[trial_numbers.notna()]
    return working


def _prepare_sst(frame: pd.DataFrame) -> pd.DataFrame:
    working = _as_ddm_trials(frame)
    ssd = pd.to_numeric(working["ssrt_or_ssd"], errors="coerce")
    return working.loc[ssd.isna()].copy()


def _prepare_cpt(frame: pd.DataFrame) -> pd.DataFrame:
    working = _as_ddm_trials(frame)
    actual_response = working["key"].map(normalize_text)
    stimulus = working["answer"].map(normalize_bool)
    working = working.loc[actual_response.notna() & stimulus.notna()].copy()
    stimulus = working["answer"].map(normalize_bool)
    working["stimulus_type"] = np.where(stimulus.astype(bool), "target", "nontarget")
    working["time_bin"] = _time_bins(working)
    return working


def _prepare_nback(trials: pd.DataFrame, domain: str) -> pd.DataFrame:
    tasks = NBACK_DOMAIN_TASKS[domain]
    working = _as_ddm_trials(trials.loc[trials["task"].isin(tasks)])
    working["load"] = working["task"].map(_nback_load)
    working = working.loc[working["load"].notna()].copy()
    working["load"] = "back" + working["load"].astype(int).astype(str)
    working["domain"] = domain
    return working


def _prepare_switch(frame: pd.DataFrame, spec: SSMModelSpec, config: dict[str, Any]) -> pd.DataFrame:
    registry = build_task_registry(config)
    task_spec = registry[spec.task or ""]
    mixed_from = int(task_spec.mixed_from or 1)
    working = _as_ddm_trials(frame)
    working["rule"] = [
        _switch_rule(str(spec.task), answer) for answer in working["answer"]
    ]
    trial_numbers = _trial_numbers(working)
    working["block"] = np.where(trial_numbers < mixed_from, "pure", "mixed")
    working["trial_type"] = _switch_trial_type(working, mixed_from)
    if spec.family == "switch_trial_type":
        working = working.loc[
            trial_numbers.ge(mixed_from) & working["trial_type"].notna()
        ].copy()
    return working.loc[working["rule"].notna()].copy()


def _prepare_stroop_race(frame: pd.DataFrame, spec: SSMModelSpec, config: dict[str, Any]) -> pd.DataFrame:
    if spec.choices is None:
        raise ValueError(f"Race model {spec.name} requires choices.")
    working = frame.copy()
    if spec.task == "ColorStroop":
        working["congruency"] = working["item"].map(conflict._color_stroop_condition)
    elif spec.task == "EmotionStroop":
        mapping = {
            str(key): value
            for key, value in config.get("emotion_stroop_conditions", {}).items()
            if value in {"congruent", "incongruent"}
        }
        working["congruency"] = working["item"].map(mapping)
    else:
        raise ValueError(f"Unsupported race task: {spec.task}")

    normalized_choices = {key.casefold(): value for key, value in spec.choices.items()}
    working["response"] = working["key"].map(
        lambda value: normalized_choices.get(str(value).casefold())
        if normalize_text(value) is not None
        else pd.NA
    )
    working["choice"] = working["response"]
    return working.loc[working["congruency"].notna() & working["response"].notna()].copy()


def build_ssm_trials(
    trials: pd.DataFrame, config: dict[str, Any], model_name: str
) -> pd.DataFrame:
    spec = get_model_spec(config, model_name)
    rt_valid = _base_rt_valid(trials, config)
    if spec.task is not None:
        task_frame = rt_valid.loc[rt_valid["task"].eq(spec.task)].copy()
    else:
        task_frame = rt_valid

    if spec.family == "flanker":
        model_trials = _prepare_flanker(task_frame)
    elif spec.family in {"dccs_overall", "dccs_block"}:
        model_trials = _prepare_dccs(task_frame, spec.family)
    elif spec.family == "sst_go_only":
        model_trials = _prepare_sst(task_frame)
    elif spec.family == "cpt_response":
        model_trials = _prepare_cpt(task_frame)
    elif spec.family == "nback_domain":
        if spec.domain is None:
            raise ValueError(f"N-back model {model_name} requires a domain.")
        model_trials = _prepare_nback(rt_valid, spec.domain)
    elif spec.family in {"switch_mixing", "switch_trial_type"}:
        model_trials = _prepare_switch(task_frame, spec, config)
    elif spec.family == "stroop_race4":
        model_trials = _prepare_stroop_race(task_frame, spec, config)
    else:
        raise ValueError(f"Unsupported SSM model family: {spec.family}")

    model_trials = model_trials.copy()
    model_trials["model_name"] = model_name
    model_trials["response"] = pd.to_numeric(model_trials["response"], errors="coerce")
    output_columns = [
        "model_name",
        "dataset",
        "subject_code",
        "task",
        "trial_index",
        "rt",
        "response",
        "choice",
        "answer",
        "key",
        "correct_trial",
        "congruency",
        "block",
        "trial_type",
        "rule",
        "stimulus_type",
        "time_bin",
        "load",
        "domain",
    ]
    for column in output_columns:
        if column not in model_trials.columns:
            model_trials[column] = pd.NA
    return model_trials.loc[:, output_columns].reset_index(drop=True)


def _formula_variables(formula: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    return tokens - {"C"}


def _formula_is_estimable(model_trials: pd.DataFrame, formula: str) -> bool:
    for variable in _formula_variables(formula):
        if variable not in model_trials.columns:
            return False
        if model_trials[variable].dropna().nunique() < 2:
            return False
    return True


def _with_random_intercept(formula: str, group_column: str = "subject_code") -> str:
    clean_formula = formula.strip()
    if not clean_formula or clean_formula == "1":
        return f"1 + (1|{group_column})"
    if f"|{group_column}" in clean_formula:
        return clean_formula
    return f"{clean_formula} + (1|{group_column})"


def effective_formulas(
    model_trials: pd.DataFrame, spec: SSMModelSpec, group_column: str = "subject_code"
) -> dict[str, str]:
    formulas = {
        parameter: formula
        for parameter, formula in spec.formulas.items()
        if _formula_is_estimable(model_trials, str(formula))
    }
    if spec.hssm_model == "ddm" and "v" not in formulas:
        formulas["v"] = "1"
    return {
        parameter: _with_random_intercept(str(formula), group_column)
        for parameter, formula in formulas.items()
    }


def trial_qc(model_trials: pd.DataFrame, spec: SSMModelSpec) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if model_trials.empty:
        return pd.DataFrame(
            [
                {
                    "model_name": spec.name,
                    "status": "trial_qc_fail",
                    "qc_reason": "no_model_trials",
                    "n_trials": 0,
                    "n_subjects": 0,
                    "n_responses": 0,
                    "n_tasks": 0,
                }
            ]
        )

    n_responses = int(model_trials["response"].nunique(dropna=True))
    status = "ok"
    reason = ""
    if len(model_trials) < 4:
        status = "trial_qc_fail"
        reason = "too_few_trials"
    elif n_responses < 2:
        status = "trial_qc_warn"
        reason = "single_response_level"

    records.append(
        {
            "model_name": spec.name,
            "status": status,
            "qc_reason": reason,
            "n_trials": len(model_trials),
            "n_subjects": int(model_trials["subject_code"].nunique(dropna=True)),
            "n_responses": n_responses,
            "n_tasks": int(model_trials["task"].nunique(dropna=True)),
        }
    )
    for column in (
        "congruency",
        "block",
        "trial_type",
        "rule",
        "stimulus_type",
        "time_bin",
        "load",
        "domain",
    ):
        if column in model_trials.columns and model_trials[column].notna().any():
            counts = model_trials[column].value_counts(dropna=False)
            for level, count in counts.items():
                records.append(
                    {
                        "model_name": spec.name,
                        "status": "condition_count",
                        "qc_reason": column,
                        "n_trials": int(count),
                        "n_subjects": pd.NA,
                        "n_responses": pd.NA,
                        "n_tasks": pd.NA,
                        "level": level,
                    }
                )
    return pd.DataFrame.from_records(records)


def p_outlier_setting(config: dict[str, Any]) -> Any:
    value = get_ssm_config(config).get("p_outlier", 0.05)
    if isinstance(value, dict):
        if value.get("estimate"):
            return dict(value["prior"])
        return float(value.get("default", 0.05))
    return float(value)


def p_outlier_default(config: dict[str, Any]) -> float:
    value = get_ssm_config(config).get("p_outlier", 0.05)
    if isinstance(value, dict):
        return float(value.get("default", 0.05))
    return float(value)


def fit_ssm_model(
    model_trials: pd.DataFrame,
    model_spec: SSMModelSpec,
    sampling: dict[str, Any],
    *,
    p_outlier: Any,
):
    import hssm

    formulas = effective_formulas(model_trials, model_spec)
    include = [
        {"name": parameter, "formula": f"{parameter} ~ {formula}"}
        for parameter, formula in formulas.items()
    ]
    hssm_data = model_trials.dropna(subset=["rt", "response"]).copy()
    hssm_data["response"] = hssm_data["response"].astype(int)
    model = hssm.HSSM(
        data=hssm_data,
        model=model_spec.hssm_model,
        include=include,
        p_outlier=p_outlier,
    )
    idata = model.sample(
        sampler=sampling.get("sampler", "nuts_numpyro"),
        draws=int(sampling.get("draws", 40)),
        tune=int(sampling.get("tune", 40)),
        chains=int(sampling.get("chains", 1)),
        random_seed=int(sampling.get("seed", 1)),
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )
    return model, idata


def summarize_ssm(
    idata: az.InferenceData,
    model_trials: pd.DataFrame,
    model_spec: SSMModelSpec,
    *,
    p_outlier: Any,
) -> pd.DataFrame:
    summary = az.summary(idata, kind="all").reset_index(names="parameter")
    summary.insert(0, "model_name", model_spec.name)
    summary["p_outlier_setting"] = str(p_outlier)
    summary["n_trials"] = len(model_trials)
    summary["n_subjects"] = int(model_trials["subject_code"].nunique(dropna=True))
    summary["n_divergences"] = _divergence_count(idata)
    summary["diagnostic_status"] = _diagnostic_status(summary, idata)
    if "mean" in summary.columns:
        summary["p_gt_0"] = [
            _posterior_probability_gt_zero(idata, parameter)
            for parameter in summary["parameter"]
        ]
    return summary


def _divergence_count(idata: az.InferenceData) -> int:
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        return 0
    return int(np.asarray(idata.sample_stats["diverging"]).sum())


def _diagnostic_status(summary: pd.DataFrame, idata: az.InferenceData) -> str:
    if _divergence_count(idata) > 0:
        return "diagnostic_fail"
    if "r_hat" not in summary.columns:
        return "diagnostic_unknown"
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    if r_hat.empty:
        return "diagnostic_unknown"
    if r_hat.gt(1.01).any():
        return "diagnostic_fail"
    return "ok"


def _posterior_probability_gt_zero(idata: az.InferenceData, parameter: str) -> float:
    if parameter not in idata.posterior:
        return float("nan")
    values = np.asarray(idata.posterior[parameter]).reshape(-1)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > 0))


def write_trace(idata: az.InferenceData, path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(resolved)
