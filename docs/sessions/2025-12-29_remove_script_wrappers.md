# 2025-12-29 remove script wrappers

## Summary
- Removed redundant `scripts/run_*` entry points, keeping only HPC scripts and `run_single_task.py`.
- Updated `README.md` and `docs/workflow.md` to reference direct `src` module entry points.

## Rationale
- Reduces `scripts/` clutter while preserving core HPC entry points.
