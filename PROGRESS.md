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
- Implemented SLURM-ready hierarchical SSM runners that save posterior traces and summaries under `data/processed/table/metrics/ssm/`: `scripts/fit_ssm_task.py` (2AFC DDM tasks) and `scripts/fit_race4_task.py` (4-choice Stroop tasks), plus collection utility `scripts/collect_ssm_results.py`.
- Added task-fMRI XCP-D task-regression pipeline (xcpd-0.7.1rc5) using `--custom_confounds` with task regressors built from Psychopy logs: `scripts/build_task_xcpd_confounds.py`, `src/imaging_preprocess/xcpd_task_36p_taskreg.sh`, and `src/imaging_preprocess/batch_run_xcpd_task.sh`; documented usage in `docs/workflow.md` and `docs/methods.md`.
- Made the task-fMRI runner robust to missing/unwritable `dataset_description.json` in fMRIPrep roots by creating a temporary wrapper root and bind-mounting `sub-<label>` into it.
- Added a task-fMRI xcp-d workflow report: `docs/reports/task_fmri_xcpd_pipeline.md`.
- Updated SST block/state regressors to support both 120-trial (single block) and 180-trial (two-block with inter-block fixation) variants, preferring `Trial_loop_list` loop identifiers when available.
- Upgraded xcp-d execution to `xcp_d-0.10.0` and aligned task-regression injection with the official `--datasets custom=...` + YAML confounds config pattern; added FreeSurfer subjects dir binding.
- Kept the xcp-d parameter set minimal (no explicit output-type/censoring flags); focused changes on container path updates and bind mounts.
- Set `--fd-thresh 100` (nonnegative) to effectively disable scrubbing under `xcp_d-0.10.0`.
- Disabled smoothing and despiking (`--smoothing 0`, `--despike n`) so task-fMRI processing matches the rest pipeline apart from task-regression confounds.
- Updated rest/task-fMRI runner defaults and documentation to CIFTI output (`--file-format cifti`) and `--min-coverage 0`.
- Unified rest/task xcp-d flags to a validated reference preset: `--output-type censored`, `--create-matrices all`, `--head-radius 50`, `--bpf-order 2`, `--resource-monitor`, and `--smoothing 2` (kept `--despike n`).
- Added a generator for task-fMRI subject lists from Psychopy logs: `scripts/build_taskfmri_sublist.py` (writes `data/processed/table/sublist/taskfmri_sublist.txt`).
- Standardized task-fMRI subject IDs in `taskfmri_sublist.txt` to the BIDS participant label format without `sub-` (required by xcp-d `--participant_label`).
- Added self-contained, shareable task-fMRI xcp-d direct scripts with hard-coded paths: `temp/run_xcpd_task_direct.sh` and `temp/batch_run_xcpd_task_direct.sh` (documented in `docs/workflow.md` and `docs/reports/task_fmri_xcpd_pipeline.md`).
- Fixed BIDS-derivatives validation for task custom confounds by adding `GeneratedBy.Name` to `dataset_description.json` (prevents `BIDSDerivativesValidationError` in xcp-d/pybids).
- Fixed a task-fMRI runner parameter regression so `src/imaging_preprocess/xcpd_task_36p_taskreg.sh` matches the unified xcp-d preset (incl. `--create-matrices all`, `--head-radius 50`, `--bpf-order 2`, `--resource-monitor`, `--smoothing 2`).
- Added an app-behavioral grouping utility (`temp/group_app_stimulus_groups.py`) that groups subjects by matching per-sheet sequences on overlapping tasks (missing sheets allowed) using the `正式阶段正确答案` column, with numeric values normalized by stripping leading zeros and an SST-specific 97-row anomaly handled by ignoring the invalid last row, and writes per-group sublists under `data/interim/behavioral_preprocess/stimulus_groups/` (config: `dataset.behavioral.interim_preprocess_dir`).


## In progress

 - Implement full-sample hierarchical DDM/LBA pipelines (including model comparisons and posterior trace saving) and SLURM submission scripts as specified in `PLAN.md`.

## Next up

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
