from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common import load_yaml, resolve_repo_path


def add_common_arguments(
    parser: argparse.ArgumentParser, *, include_limit: bool = False
) -> None:
    parser.add_argument("--dataset", default="THU")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--metrics-config", default="configs/behavioral_metrics.yaml")
    parser.add_argument("--output-dir")
    if include_limit:
        parser.add_argument("--limit", type=int)


def load_runtime_config(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    paths = load_yaml(args.paths_config)
    metrics = load_yaml(args.metrics_config)
    dataset_paths = paths["datasets"][args.dataset]
    app_data_dir = resolve_repo_path(dataset_paths["app_data_dir"])
    output_dir = resolve_repo_path(args.output_dir or dataset_paths["behavioral_metrics_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return app_data_dir, output_dir, metrics
