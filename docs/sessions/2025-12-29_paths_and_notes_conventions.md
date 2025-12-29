# Session log (2025-12-29): path conventions and docs semantics

## Scope

- Engineering changes to configuration wiring and script invocation (no changes to scientific/statistical logic).
- Documentation updates clarifying `docs/sessions/` vs `docs/notes/`.

## Changes

- Updated `configs/paths.yaml` to define per-dataset roots for `data/raw/`, `data/interim/`, `data/processed/`, and `outputs/`.
- Updated `configs/datasets/EFNY.yaml`:
  - clarified that modeling inputs are resolved relative to `data/processed/EFNY/`,
  - added `external_inputs.fmriprep_dir` to support fMRIPrep outputs outside the repo.
- Updated config resolution to expose `raw_root`, `interim_root`, and `processed_root` in the standardized entry point (`scripts/run_single_task.py`).
- Updated `PROJECT_STRUCTURE.md` and `AGENTS.md` to define:
  - `docs/sessions/` as session logs,
  - `docs/notes/` as user notes and free-form ideas.

## Verification

- `python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run`

