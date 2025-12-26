# Session Record: fix import path in missing metrics script

## Summary
- Fixed module import path handling in the missing-metrics checker so it runs from `src`.

## Actions Taken
- Added `sys.path` bootstrapping in `src/app_data_proc/check_missing_beh_metrics.py` and switched to local package import.

## Artifacts
- `src/app_data_proc/check_missing_beh_metrics.py`: updated import path handling.
