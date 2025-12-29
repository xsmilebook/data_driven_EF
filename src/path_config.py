from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_io import load_simple_yaml


def load_paths_config(paths_yaml: str | Path, *, repo_root: Path) -> dict[str, Any]:
    cfg = load_simple_yaml(paths_yaml)
    cfg["__repo_root__"] = str(repo_root)
    return cfg


def resolve_dataset_roots(paths_cfg: dict[str, Any], *, dataset: str) -> dict[str, Path]:
    repo_root = Path(paths_cfg.get("__repo_root__", ".")).resolve()
    datasets = paths_cfg.get("datasets", {})
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise KeyError(f"Dataset not found in paths config: {dataset}")

    ds = datasets[dataset]
    if not isinstance(ds, dict):
        raise ValueError(f"Invalid dataset config for {dataset}: expected mapping")

    raw_root = _resolve_path(repo_root, ds.get("raw_root"))
    interim_root = _resolve_path(repo_root, ds.get("interim_root"))
    processed_root = _resolve_path(repo_root, ds.get("processed_root"))
    outputs_root = _resolve_path(repo_root, ds.get("outputs_root"))

    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "interim_root": interim_root,
        "processed_root": processed_root,
        "outputs_root": outputs_root,
    }


def _resolve_path(repo_root: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Missing required path value")
    p = Path(str(value))
    return p if p.is_absolute() else (repo_root / p)

