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
- Updated `docs/reports/ddm_decision.md` and `docs/workflow.md` to reflect finalized model eligibility: 4-choice LBA for Stroops, 2-choice recoding for DT/EmotionSwitch, and go-only DDM for SST; updated `PLAN.md` accordingly.
- Removed the Emotion1Back/Emotion2Back “reference item-column replacement” from `src/behavioral_preprocess/app_data/format_app_data.py` (kept only the SST SSRT reference fix).
- Updated `docs/reports/ddm_decision.md` and `PLAN.md` to specify two parallel hierarchical DDMs for DT/EmotionSwitch (Mixing vs Switch), including condition effects on `v/a/t0` with `rule` interactions.


## In progress

 - Implement full-sample hierarchical DDM/LBA pipelines (including model comparisons and posterior trace saving) and SLURM submission scripts as specified in `PLAN.md`.

## Next up

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
