# PROGRESS.md

Real-time development log for the refactor of `data_driven_EF`.

## Current focus

Engineering-only refactor to standardize paths, entry points, and documentation without changing scientific logic.

## Completed

- Introduced config-driven path resolution and dataset roots (`configs/paths.yaml`, `configs/datasets/EFNY.yaml`).
- Standardized raw/interim/processed/output path conventions across multiple scripts.
- Added `scripts/render_paths.py` for SLURM/bash scripts to consume config-derived roots.
- Added dataset-agnostic `configs/analysis.yaml` and wired defaults into `scripts/run_single_task.py`.
- Removed residual `sys.path` hacks in `src/` where feasible.
- Documented sessions and updated `docs/workflow.md`, `docs/README.md`, and `docs/data_dictionary.md`.
- Standardized result summary scripts to infer `results_root` from `--dataset/--config` with default `outputs/<DATASET>/results`.
- Updated `README.md` to reflect standardized CLI and config usage.
- Updated `PLAN.md` with atlas selection guidance for Schaefer100/200/400 via FC matrix path patterns.
- Documented atlas selection guidance in `configs/datasets/EFNY.yaml` and `docs/datasets/EFNY.md`.
- Added config-driven defaults to `src/preprocess/screen_head_motion_efny.py` and a thin entry point `scripts/run_screen_head_motion.py`.
- Added config-driven defaults and thin entry points for app-data processing utilities (behavioral data build, task analysis, missing-metrics report).
- Added config-driven defaults for demo preprocessing and behavioral metric exploration outputs.
- Added thin entry points for demo preprocessing and behavioral metric exploration.
- Removed hard-coded app-data input directory defaults from format_app_data.
- Switched functional-connection SLURM scripts to use config-derived output root for log directories.
- Updated functional-connection plotting example to use `outputs/<DATASET>` path.
- Added notes in SLURM scripts explaining static SBATCH log paths.
- Added config-driven defaults for `compute_fc_schaefer.py` and removed EFNY-specific defaults.
- Updated README usage for `compute_fc_schaefer.py` to include dataset/config defaults.
- Updated workflow and EFNY dataset docs to reflect `data/raw`/`data/interim`/`data/processed`/`outputs` conventions.
- Added README note about static SBATCH log paths.
- Added config-driven defaults for metrics similarity heatmaps, EFNY metrics computation, group-average FC, Fisher-Z FC, and vectorization.
- Added thin `scripts/` entry points for remaining runnable utilities (functional_conn, metric_compute, result_summary, app_data_proc).
- Updated README to reflect new standardized CLI usage for FC and behavioral tools.
- Updated `docs/workflow.md` to recommend `scripts.run_*` wrappers.
- Added README usage blocks for result_summary script wrappers.
- Added example CLI blocks to `docs/workflow.md`.
- Removed legacy JSON config support (`config.json` / `--config_file`) from modeling utilities.
- Removed redundant `scripts/run_*` wrappers; documentation now points directly to `src` entry points.
- Moved CV hyperparameter defaults into `configs/analysis.yaml` and wired them into `run_single_task.py`.
- Added additional evaluation/preprocessing defaults (PCA, standardize, shuffle seeds, score metric, permutation iterations) and passed into nested CV.
- Ran `py_compile` and `run_single_task --dry-run` after the updates.

## In progress

- None.

## Next up

- Add thin `scripts/` entry points for other frequently used tools as needed.
- Ensure remaining preprocessing and FC scripts read external input roots from configs where applicable.
- Run `py_compile` and dry-run checks after each batch of changes.

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|
| Mixed output roots (legacy data root vs outputs) | Normalize to `data/raw`, `data/interim`, `data/processed`, `outputs` | 2025-12-29 |
| Hard-coded cluster paths in bash scripts | Use `scripts/render_paths.py` to inject config-derived paths | 2025-12-29 |

## Notes

- Only refactor scope is active; feature development resumes after refactor completion.
- Keep session logs under `docs/sessions/` for each AI-assisted change set.
