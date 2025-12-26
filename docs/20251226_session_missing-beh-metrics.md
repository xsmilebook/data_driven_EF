# Session Record: missing behavioral metrics check

## Summary
- Added a script to report subjects with existing task sheets but missing behavioral metrics in the EFNY metrics table.
- Relaxed trial-prep gating so metrics can still be computed when some trials are filtered out.

## Actions Taken
- Added `src/app_data_proc/check_missing_beh_metrics.py` to scan sheets vs. metrics and write a report under the results tree.
- Updated `src/metric_compute/efny/preprocess.py` to allow downstream metrics to run even when low trial proportions are detected.

## Artifacts
- `src/app_data_proc/check_missing_beh_metrics.py`: missing-metrics report generator.
- `src/metric_compute/efny/preprocess.py`: low-proportion handling in trial preparation.

## Next Steps
- Run the new check script to generate the report and review any high-missing tasks.
