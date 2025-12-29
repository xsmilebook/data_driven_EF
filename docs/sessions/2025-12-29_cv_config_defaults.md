# 2025-12-29 CV config defaults

## Summary
- Added CV-related defaults to `configs/analysis.yaml`.
- Wired `run_single_task.py` to read outer/inner CV shuffle and inner split counts from config.

## Rationale
- Centralizes CV hyperparameters in configuration instead of hard-coding them.
