#!/usr/bin/env python3
"""
Render resolved dataset paths from configs for shell usage.

Intended use (cluster bash scripts):
  eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_io import load_simple_yaml
from src.path_config import load_paths_config, resolve_dataset_roots


def _shell_escape_double_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _bash_export(name: str, value: str) -> str:
    return f'export {name}="{_shell_escape_double_quotes(value)}"'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", dest="paths_config", default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", default=None)
    ap.add_argument("--format", choices=["bash"], default="bash")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)

    dataset_cfg_path = (
        Path(args.dataset_config)
        if args.dataset_config is not None
        else (repo_root / "configs" / "datasets" / f"{args.dataset}.yaml")
    )
    dataset_cfg = load_simple_yaml(dataset_cfg_path)
    external_inputs = dataset_cfg.get("external_inputs", {})
    fmriprep_dir = ""
    if isinstance(external_inputs, dict):
        fmriprep_dir = str(external_inputs.get("fmriprep_dir", "") or "")

    lines: list[str] = []
    lines.append(_bash_export("PROJECT_DIR", str(roots["repo_root"])))
    lines.append(_bash_export("DATASET", str(args.dataset)))
    lines.append(_bash_export("RAW_ROOT", str(roots["raw_root"])))
    lines.append(_bash_export("INTERIM_ROOT", str(roots["interim_root"])))
    lines.append(_bash_export("PROCESSED_ROOT", str(roots["processed_root"])))
    lines.append(_bash_export("OUTPUTS_ROOT", str(roots["outputs_root"])))
    lines.append(_bash_export("LOGS_ROOT", str(roots["logs_root"])))
    if fmriprep_dir:
        lines.append(_bash_export("FMRIPREP_DIR", fmriprep_dir))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
