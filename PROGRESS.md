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
- Added EFNY-XY (Xiangya) task-fMRI support via `configs/dataset_xiangya_taskfmri.yaml` and dataset-config aware runners that save outputs under `data/interim/MRI_data/xcpd_task_xy/`.
- Updated `scripts/build_taskfmri_sublist.py` to support BIDS-style XY folders named `sub-<LABEL>` (writes `<LABEL>` to the sublist so xcp-d can match `--participant_label`).
- Added copy-paste commands for THU/XY task-fMRI sublist generation and SLURM submission to `docs/workflow.md`.
- Standardized dataset labels to `EFNY_THU` and `EFNY_XY` for clearer CLI usage and log separation.
- Updated XY task-fMRI sublist generation to convert Psychopy folder names like `XY_YYYYMMDD_NUM_CODE` to participant labels like `XY<YYYYMMDD><NUM><CODE>` (underscores removed) to match fMRIPrep `sub-XY...`.
- Fixed XY task confounds discovery so `scripts/build_task_xcpd_confounds.py` can locate behavior CSVs under `task_psych_xy/XY_.../`, preventing missing `/custom_confounds/confounds_config.yml` at xcp-d runtime.
- Made `src/imaging_preprocess/batch_run_xcpd_task.sh` skip already-successful subject×task outputs by default (override with `XCPD_FORCE=1`).
- Fixed XY Psychopy compatibility in `scripts/build_task_xcpd_confounds.py` by detecting trial rows and block timing from task-specific columns (e.g. `Trial_text.started`, `Trial_image_1.started`) instead of requiring `Trial.started`.
- Added an app-behavioral grouping utility (`temp/group_app_stimulus_groups.py`) that groups subjects by matching per-sheet sequences on overlapping tasks (missing sheets allowed) for either `正式阶段正确答案` (`--grouping answer`) or `正式阶段刺激图片/Item名` (`--grouping item`), with numeric values normalized by stripping leading zeros and an SST-specific 97-row anomaly handled by ignoring the invalid last row; outputs are written under `data/interim/behavioral_preprocess/groups_by_answer/` and `data/interim/behavioral_preprocess/groups_by_item/`.
- Added a repository rule to remove smoke-test outputs under `temp/` immediately after validation to avoid leaving ad-hoc artifacts.
- Added a detailed app behavioral data correction plan to `PLAN.md` (infer corrected stimulus+answer sequences across visit versions without overwriting raw workbooks).
- Added a visit-config-based app workbook corrector (`scripts/correct_app_workbooks.py`) that writes per-subject corrected Excel files to `data/processed/behavior_data/cibr_app_data_corrected_excel/` and logs per-task diagnostics; documented in `docs/reports/app_sequence_correction_report.md`.
- Added a heuristic classifier for whether a workbook was generated from device-side txt export vs web export using the presence of the `空屏时长` column in all task sheets; wrote `export_source` evidence into `manifest.csv` and `decisions/*.json` for each correction run.
- Expanded `docs/reports/app_sequence_correction_report.md` with a detailed, auditable description of the inference+correction method and clarified the current limitation around visit3 sub-versions (snapshots not yet available in `app_sequence/visit3/`).
- Archived duplicate APP `*_GameData.xlsx` workbooks from `data/raw/behavior_data/cibr_app_data/` according to a manual keep-list (moved non-kept files into `data/raw/behavior_data/cibr_app_data/_excluded_duplicates/` via `temp/cleanup_app_duplicate_workbooks.py`).
- Added a scanner for repeated task Psychopy CSV logs per subject×task: `temp/find_repeated_task_psych_csvs.py` (report: `data/interim/MRI_data/repeated_task_psych_file/repeated_task_psych_files.tsv`).
- Switched task-fMRI behavior CSV selection to use the filename timestamp (`YYYY-MM-DD_HHhMM.SS.mmm`) when multiple logs exist for a subject×task.
- Standardized task-fMRI behavior data documentation and added SST 180-trial and THU/XY Psychopy-format notes: `docs/reports/task_fmri_beh_data_handbook.md`.
- Adjusted task-fMRI block/state timing to use stimulus presentation columns (not fixation) in `scripts/build_task_xcpd_confounds.py`.
- Hard-coded proxy envvars (with scheme, e.g. `http://host:port`) for xcp-d runs and passed them into Singularity via `SINGULARITYENV_*` to avoid `ProxySchemeUnknown` errors on compute nodes.
- Removed duplicate APP corrected outputs in `run_corrected_v1` to match `_excluded_duplicates`, updating manifests accordingly.
- Analyzed 2025-12-07/12-13 low-score APP subjects and documented visit-mismatch evidence in `docs/notes/app_data_problem.md`.
- Added a consolidated APP visit sequence change note covering visit1–visit4 differences in `docs/reports/visit_sequence_changes.md`.
- Audited 20231014–20250707 APP subjects for visit1 consistency and documented task/subject mismatches in `docs/reports/visit1_consistency_20231014_20250707.md`.
- Audited existing app sequence grouping outputs under `data/interim/behavioral_preprocess/` and drafted a v2 cleaning workflow in `docs/reports/app_data_cleaning_v2_workflow.md`.
- Archived prior app grouping outputs and regenerated item/answer groupings from the current raw directory (updated `docs/reports/app_grouping_audit.md`).
- Built a v2 visit1 sequence library using item group_001 templates plus `app_sequence/visit1` answers under `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/`.
- Added a checker to compare group templates vs `app_sequence` items and wrote a visit1 consistency report for items: `docs/reports/visit1_items_template_vs_app_sequence.md`.
- Documented v2 item handling exceptions for KT/ZYST/SST and added a visit1 answer-group audit within item group_001: `docs/reports/visit1_answer_groups_within_item_group1.md`.
- Applied v2 answer-side fixes (SST truncation to 96 trials; EmotionStroop trial-13 fill with AN when otherwise matches) and merged answer group_003/group_005/group_006 back into group_001 (audit: `docs/reports/app_v2_answer_group_merge.md`).
- Added an item-group (group_001–group_004) sequence-difference report to guide v2 item-based diagnosis (updated to use 18-sheet representatives and include KT/ZYST/SST): `docs/reports/item_group_sequence_changes.md`.
- Corrected v2 item-evidence rules to include KT and FZSS (updated `docs/reports/app_data_cleaning_v2_workflow.md`, `docs/reports/app_sequence_correction_report.md`, and `docs/reports/visit_sequence_changes.md`).
- Updated v2 sequence-library policy to not treat app_sequence-derived answers as ground truth; visit1_merged now aligns items to group_001 templates and uses template-derived observed answers (app_sequence kept as reference only).
- Replaced item sequences for answer_group1 subjects that mismatched item_group1 by overwriting workbook item columns with item_group1 templates (website-crash reference-subject error), and regenerated item groupings (report: `docs/reports/app_v2_item_replace_answer_group1.md`; script: `scripts/app_v2_replace_items_for_answer_group_mismatch.py`).
- Translated `AGENTS.md` into Chinese and added a Windows dependency-management rule: use `uv` with `uv add`, and keep the project virtual environment at `.venv`.
- Exported a visit3 candidate sequence in v2 from `item_group_002` (items) + `answer_group_002` (answers), and compared it against `data/raw/behavior_data/app_sequence/visit3` (report: `docs/reports/visit3_group2_export_comparison.md`; script: `scripts/build_visit3_library_v2.py`).
- Parsed non-standard `visit3_v1` Excel files into JSON and compared to `answer_group_004` for Spatial/Emotion NBack tasks (report: `docs/reports/visit3_v1_sequence_parsing.md`; script: `scripts/parse_visit3_v1_sequence.py`).
- Built `visit3_v1` in `run_corrected_v2` from confirmed rules: keep visit3 baseline for non-target tasks, replace Spatial1Back/Spatial2Back from `app_sequence/visit3_v1`, and set Emotion2Back items to null with answers from `answer_group_004` (report: `docs/reports/visit3_v1_from_groups_build.md`; script: `scripts/build_visit3_v1_library_v2.py`).
- Fixed visit3_v1 sequence export to preserve full trial length (including leading missing answers): Emotion2Back is now 60/60 (items/answers) and row metadata is synchronized across tasks.
- Applied `visit3_v1` item templates to all `answer_group_004` subjects (36/36) with workbook backup, then moved these subjects from item `group_002` to new item `group_005`; regenerated item-group manifests (report: `docs/reports/app_v2_item_replace_answer_group004.md`; script: `scripts/app_v2_apply_visit3_v1_items_to_answer_group004.py`).
- Updated `THU_20250921_765_WZR` item sequence to `visit3` (because answer group is 002), with a per-subject backup and execution log documented in `docs/reports/app_v2_item_replace_subject_765_wzr.md` (script: `scripts/app_v2_apply_visit_items_to_subject.py`).
- Moved `THU_20250921_765_WZR` from item `group_004` to item `group_002`, and regenerated `groups_by_item` manifests to keep counts/index fields synchronized (`group_002: 139`, `group_004: 0`).
- Added a conformed export step for confirmed `run_corrected_v2` sequence JSONs: `visit1_confirmed.json`, `visit3_confirmed.json`, `visit3_v1_confirmed.json` under `run_corrected_v2/conformed/`, with source mapping in `conformed/manifest.json` (script: `scripts/export_conformed_sequences_v2.py`).


## In progress

 - Implement full-sample hierarchical DDM/LBA pipelines (including model comparisons and posterior trace saving) and SLURM submission scripts as specified in `PLAN.md`.

## Next up

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
