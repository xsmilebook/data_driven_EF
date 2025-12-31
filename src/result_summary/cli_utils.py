from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


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
    results_root is set to outputs/results.
    """
    if results_root:
        return Path(results_root)

    if not dataset:
        raise ValueError("Missing --dataset (required when --results_root is not provided).")

    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=dataset)

    _ = load_dataset_config(
        paths_cfg,
        dataset_config_path=dataset_config,
        repo_root=repo_root,
    )

    return roots["outputs_root"] / "results"
