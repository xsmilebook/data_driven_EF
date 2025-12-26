# Session Record: fix n-back missing metrics fallback

## Summary
- Added fallback trial-type classification for number/spatial n-back tasks when item values are missing.
- Regenerated EFNY behavioral metrics and the missing-metrics report.

## Actions Taken
- Updated `src/metric_compute/efny/metrics.py` to infer target/nontarget from answer when item is empty.
- Added answer-mapping config for number/spatial n-back tasks in `src/metric_compute/efny/main.py`.
- Ran `python src/metric_compute/compute_efny_metrics.py`.
- Ran `python src/app_data_proc/check_missing_beh_metrics.py`.

## Artifacts
- `src/metric_compute/efny/metrics.py`: fallback trial typing for n-back.
- `src/metric_compute/efny/main.py`: target/nontarget answer mapping.
- `data/EFNY/table/metrics/EFNY_beh_metrics.csv`: regenerated metrics.
- `data/EFNY/results/behavior_data/behavior_metrics_missing_report.csv`: regenerated report.
