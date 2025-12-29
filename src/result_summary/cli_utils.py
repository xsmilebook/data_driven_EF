from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config_io import load_simple_yaml
from src.path_config import load_paths_config, resolve_dataset_roots


def resolve_results_root(
    *,
    results_root: Optional[str],
    dataset: Optional[str],
    paths_config: str,
    dataset_config: Optional[str],
) -> Path:
    """
    Resolve results_root from CLI args and centralized configs.

    If results_root is provided, it wins. Otherwise, dataset is required and
    results_root is set to outputs/<DATASET>/results.
    """
    if results_root:
        return Path(results_root)

    if not dataset:
        raise ValueError("Missing --dataset (required when --results_root is not provided).")

    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=dataset)

    if dataset_config:
        _ = load_simple_yaml(dataset_config)
    else:
        _ = load_simple_yaml(repo_root / "configs" / "datasets" / f"{dataset}.yaml")

    return roots["outputs_root"] / "results"

