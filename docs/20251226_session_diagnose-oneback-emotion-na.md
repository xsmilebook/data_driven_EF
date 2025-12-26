# Session Record: diagnose oneback_emotion NA

## Summary
- Added a temporary diagnostic script to explain why oneback_emotion metrics are NA.

## Actions Taken
- Created `src/app_data_proc/diagnose_oneback_emotion_na.py` to report per-subject reasons and counts.

## Artifacts
- `src/app_data_proc/diagnose_oneback_emotion_na.py`

## Updates
- Added sys.path bootstrapping to `src/app_data_proc/diagnose_oneback_emotion_na.py` so it runs from `src`.
- Ran the diagnostic script; output saved to `data/EFNY/results/behavior_data/oneback_emotion_na_diagnosis.csv`.
