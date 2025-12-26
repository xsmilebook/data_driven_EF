# Session Record: remove TASK_CSV dependency

## Summary
- Updated EFNY metric computation entrypoint to use built-in task config instead of the task CSV.

## Actions Taken
- Switched `src/metric_compute/compute_efny_metrics.py` to call `run_raw` and removed unused imports/constants.

## Artifacts
- `src/metric_compute/compute_efny_metrics.py`: simplified to run without `EFNY_task.csv`.
