from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class V2PipelineConfig:
    dataset: str
    behavior_table: Path
    output_dir: Path
    subject_candidates: tuple[str, ...] = ("subject_code", "subid", "id")


def resolve_behavior_table(
    *,
    behavior_table_arg: str | None,
    dataset_cfg: dict[str, Any],
    roots: dict[str, Path],
) -> Path:
    if behavior_table_arg:
        p = Path(behavior_table_arg)
        if p.is_absolute():
            return p
        return (roots["repo_root"] / p).resolve()

    files_cfg = dataset_cfg.get("files", {})
    if not isinstance(files_cfg, dict):
        raise ValueError("Invalid dataset config: files must be a mapping.")
    rel = files_cfg.get("behavioral_metrics_file")
    if not rel:
        raise ValueError("Missing dataset.files.behavioral_metrics_file in config.")
    return (roots["processed_root"] / str(rel)).resolve()


def resolve_output_dir(
    *,
    output_dir_arg: str | None,
    roots: dict[str, Path],
    run_id: str,
) -> Path:
    if output_dir_arg:
        p = Path(output_dir_arg)
        if p.is_absolute():
            return p
        return (roots["repo_root"] / p).resolve()
    return (roots["outputs_root"] / "results" / "v2" / run_id).resolve()


def choose_subject_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def profile_behavior_table(path: Path, *, subject_candidates: tuple[str, ...]) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8")
    subject_col = choose_subject_column(df, subject_candidates)

    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    n_numeric_cols = int(df.select_dtypes(include=["number"]).shape[1])
    missing_rate = float(df.isna().sum().sum() / (n_rows * n_cols)) if n_rows > 0 and n_cols > 0 else 0.0

    n_subjects = None
    if subject_col is not None:
        n_subjects = int(df[subject_col].astype(str).nunique())

    return {
        "behavior_table": str(path),
        "rows": n_rows,
        "columns": n_cols,
        "numeric_columns": n_numeric_cols,
        "missing_rate": missing_rate,
        "subject_column": subject_col,
        "subjects": n_subjects,
    }


def build_run_payload(cfg: V2PipelineConfig, *, dry_run: bool, profile: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": cfg.dataset,
        "dry_run": bool(dry_run),
        "behavior_table": str(cfg.behavior_table),
        "output_dir": str(cfg.output_dir),
        "profile": profile,
    }


def write_v2_stub_outputs(cfg: V2PipelineConfig, payload: dict[str, Any]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# v2 Pipeline Stub Output",
        "",
        "This directory is created by `python -m scripts.v2_run_pipeline`.",
        "",
        "Current scope:",
        "- Resolve config and dataset paths",
        "- Validate cleaned behavioral table readability",
        "- Emit profile metadata for migration checks",
    ]
    (cfg.output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

