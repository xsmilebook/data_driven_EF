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

## In progress

- None.

## Next up

- Add thin `scripts/` entry points for other frequently used tools as needed.
- Ensure remaining preprocessing and FC scripts read external input roots from configs where applicable.
- Run `py_compile` and dry-run checks after each batch of changes.

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|
| Mixed output roots (`data/EFNY`, `outputs/EFNY`) | Normalize to `data/raw`, `data/interim`, `data/processed`, `outputs` | 2025-12-29 |
| Hard-coded cluster paths in bash scripts | Use `scripts/render_paths.py` to inject config-derived paths | 2025-12-29 |

## Notes

- Only refactor scope is active; feature development resumes after refactor completion.
- Keep session logs under `docs/sessions/` for each AI-assisted change set.
