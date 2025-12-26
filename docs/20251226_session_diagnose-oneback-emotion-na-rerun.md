# Session Record: diagnose oneback_emotion NA (rerun)

## Summary
- Recreated and ran a diagnostic script to explain why oneback_emotion metrics are still NA after updates.

## Actions Taken
- Added `src/app_data_proc/diagnose_oneback_emotion_na.py` and saved results to `data/EFNY/results/behavior_data/oneback_emotion_na_diagnosis.csv`.

## Results
- oneback_emotion NA reasons: 212 no_target_trials, 17 prep_failed_low_prop_or_missing_rt.
