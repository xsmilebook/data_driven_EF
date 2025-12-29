# 2025-12-29 evaluation config expansion

## Summary
- Expanded `configs/analysis.yaml` with PCA, standardization, shuffle seeds, permutation iterations, and score metric defaults.
- Wired `run_single_task.py` to read these defaults and pass them into nested CV.
- Added validation for supported `score_metric` in evaluation.

## Rationale
- Centralizes evaluation and preprocessing controls in config while preserving existing logic.
