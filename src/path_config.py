from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_io import load_simple_yaml


def load_paths_config(paths_yaml: str | Path, *, repo_root: Path) -> dict[str, Any]:
    cfg = load_simple_yaml(paths_yaml)
    cfg["__repo_root__"] = str(repo_root)
    return cfg


def load_dataset_config(
    paths_cfg: dict[str, Any],
    *,
    dataset_config_path: str | Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    if dataset_config_path:
        cfg = load_simple_yaml(dataset_config_path)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid dataset config: expected mapping ({dataset_config_path})")
        return cfg

    dataset_cfg = paths_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict) or not dataset_cfg:
        raise ValueError("Missing dataset config in configs/paths.yaml (dataset section).")
    return dataset_cfg


def resolve_dataset_roots(paths_cfg: dict[str, Any], *, dataset: str | None = None) -> dict[str, Path]:
    repo_root = Path(paths_cfg.get("__repo_root__", ".")).resolve()
    datasets = paths_cfg.get("datasets", {})
    ds = None
    if isinstance(datasets, dict) and dataset is not None:
        ds = datasets.get(dataset)
        if ds is not None and not isinstance(ds, dict):
            raise ValueError(f"Invalid dataset config for {dataset}: expected mapping")

    if ds is not None:
        raw_root = _resolve_path(repo_root, ds.get("raw_root"))
        interim_root = _resolve_path(repo_root, ds.get("interim_root"))
        processed_root = _resolve_path(repo_root, ds.get("processed_root"))
        outputs_root = _resolve_path(repo_root, ds.get("outputs_root"))
    else:
        raw_root = _resolve_path(repo_root, paths_cfg.get("raw_root") or Path(paths_cfg.get("data_root", "data")) / "raw")
        interim_root = _resolve_path(repo_root, paths_cfg.get("interim_root") or Path(paths_cfg.get("data_root", "data")) / "interim")
        processed_root = _resolve_path(repo_root, paths_cfg.get("processed_root") or Path(paths_cfg.get("data_root", "data")) / "processed")
        outputs_root = _resolve_path(repo_root, paths_cfg.get("outputs_root", "outputs"))
    logs_root = _resolve_path(repo_root, paths_cfg.get("logs_root", "logs"))

    return {
        "repo_root": repo_root,
        "raw_root": raw_root,
        "interim_root": interim_root,
        "processed_root": processed_root,
        "outputs_root": outputs_root,
        "logs_root": logs_root,
    }


def _resolve_path(repo_root: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Missing required path value")
    p = Path(str(value))
    return p if p.is_absolute() else (repo_root / p)
