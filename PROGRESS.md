# PROGRESS.md

Real-time development log for the refactor of `data_driven_EF`.

## Current focus

## Completed

- Expanded behavioral metrics computation documentation in `docs/methods.md`.
- Removed `RT_Mean`/`RT_SD` outputs for SST/GNG/CPT in metrics configuration and docs.
- Added a single-subject behavioral EDA entrypoint (`scripts/eda_behavior_trials.py`) and generated a per-task trial-count report under `docs/reports/`.
- Updated the behavioral EDA report to include SST Go/Stop trial counts and to avoid RT-based filtering (single-subject robustness).
- Updated switch-task mixed-block thresholds (`mixed_from`) for DCCS/DT/EmotionSwitch and refreshed docs/report (including updated DDM decision text) accordingly.
- Added a PyMC-based hierarchical HDDM decision+fit pipeline (`scripts/ddm_decision_report.py`) and generated `docs/reports/ddm_decision.md`.


## In progress

## Next up

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
