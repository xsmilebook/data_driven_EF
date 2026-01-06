#!/usr/bin/env python3
"""
Render resolved dataset paths from configs for shell usage.

Intended use (cluster bash scripts):
  eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def _shell_escape_double_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _bash_export(name: str, value: str) -> str:
    return f'export {name}="{_shell_escape_double_quotes(value)}"'

def _resolve_maybe_relative(value: str, *, base: Path) -> str:
    s = str(value).strip()
    if s.startswith("/"):
        return s
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


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

    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    external_inputs = dataset_cfg.get("external_inputs", {})
    outputs_cfg = dataset_cfg.get("outputs", {})
    fmriprep_dir = ""
    task_psych_dir = ""
    freesurfer_subjects_dir = ""
    xcpd_task_dirname = ""
    fmriprep_task_dirs: dict[str, str] = {}
    if isinstance(external_inputs, dict):
        fmriprep_dir = str(external_inputs.get("fmriprep_dir", "") or "")
        task_psych_dir = str(external_inputs.get("task_psych_dir", "") or "")
        freesurfer_subjects_dir = str(external_inputs.get("freesurfer_subjects_dir", "") or "")
        # Accept either a mapping (recommended) or per-task keys.
        raw_task_dirs = external_inputs.get("fmriprep_task_dirs", {})
        if isinstance(raw_task_dirs, dict):
            for k, v in raw_task_dirs.items():
                if v:
                    fmriprep_task_dirs[str(k)] = str(v)
        for task in ("nback", "sst", "switch"):
            k = f"fmriprep_task_{task}_dir"
            v = external_inputs.get(k)
            if v:
                fmriprep_task_dirs[task] = str(v)

    if isinstance(outputs_cfg, dict):
        xcpd_task_dirname = str(outputs_cfg.get("xcpd_task_dirname", "") or "")

    lines: list[str] = []
    lines.append(_bash_export("PROJECT_DIR", str(roots["repo_root"])))
    lines.append(_bash_export("DATASET", str(args.dataset)))
    lines.append(_bash_export("RAW_ROOT", str(roots["raw_root"])))
    lines.append(_bash_export("INTERIM_ROOT", str(roots["interim_root"])))
    lines.append(_bash_export("PROCESSED_ROOT", str(roots["processed_root"])))
    lines.append(_bash_export("OUTPUTS_ROOT", str(roots["outputs_root"])))
    lines.append(_bash_export("LOGS_ROOT", str(roots["logs_root"])))
    if fmriprep_dir:
        lines.append(_bash_export("FMRIPREP_DIR", _resolve_maybe_relative(fmriprep_dir, base=roots["repo_root"])))
    if task_psych_dir:
        lines.append(_bash_export("TASK_PSYCH_DIR", _resolve_maybe_relative(task_psych_dir, base=roots["raw_root"])))
    if freesurfer_subjects_dir:
        lines.append(
            _bash_export("FREESURFER_SUBJECTS_DIR", _resolve_maybe_relative(freesurfer_subjects_dir, base=roots["repo_root"]))
        )
    if xcpd_task_dirname:
        lines.append(_bash_export("XCPD_TASK_DIRNAME", xcpd_task_dirname))
    for task, path in sorted(fmriprep_task_dirs.items()):
        env_name = f"FMRIPREP_TASK_{task.upper()}_DIR"
        lines.append(_bash_export(env_name, _resolve_maybe_relative(path, base=roots["repo_root"])))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
