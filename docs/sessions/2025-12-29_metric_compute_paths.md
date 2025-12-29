# 2025-12-29 metric_compute and demo preprocessing paths

## Summary
- Removed hard-coded EFNY paths from `src/metric_compute/behavioral_metric_exploration.py`.
- Added config-driven defaults to `src/preprocess/preprocess_efny_demo.py`.
- Extended `configs/datasets/EFNY.yaml` with demo/QC file entries.

## Rationale
- Keeps filesystem paths centralized in configs.
- Maintains dataset isolation and allows reuse across datasets.
