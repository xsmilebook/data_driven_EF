# Neuroimaging Prep Pipeline (EFNY)

This document summarizes the preprocessing logic and the intended run order for scripts under `src/preprocess/`.

## Run Order
1) xcpd
2) screen_head_motion
3) preprocess_efny_demo

## 1) xcpd (rest preprocessing)
Scripts:
- `src/preprocess/batch_run_xcpd.sh`
- `src/preprocess/xcpd_36p.sh`

Logic:
- `batch_run_xcpd.sh` reads subject IDs from `data/EFNY/table/sublist/mri_sublist.txt` and submits one SLURM job per subject.
- Each job runs `xcpd_36p.sh <subj>` which:
  - Loads Singularity and binds fMRIPrep outputs, working directories, FreeSurfer license, and TemplateFlow.
  - Runs `xcpd` (v0.7.1rc5) on `task-id rest` with 36P nuisance regressors, despike, band-pass 0.01�C0.1 Hz, motion low-pass filter, and no FD censoring (`--fd-thresh -1`).
  - Uses 2 threads and 40 GB memory; writes outputs to `data/EFNY/MRI_data/xcpd_rest` and removes the workdir after completion.

## 2) screen_head_motion (QC summary)
Script:
- `src/preprocess/screen_head_motion_efny.py`

Logic:
- Scans the fMRIPrep directory for `*task-rest*desc-confounds_timeseries.tsv` files under `func/`.
- For runs 1~4 only, computes:
  - frame count
  - mean FD
  - ratio of frames with FD > 0.3
- Uses all 180-frame runs to compute a global FD upper limit (Q3 + 1.5*IQR) for outlier screening.
- A run is valid if: frame count = 180, high-FD ratio <= 0.25, and mean FD <= global upper limit.
- A subject is valid if at least two runs are valid; also stores mean FD across valid runs.
- Writes `rest_fd_summary.csv` with per-run stats, validity flags, and `meanFD`.

## 3) preprocess_efny_demo (demo + QC merge)
Script:
- `src/preprocess/preprocess_efny_demo.py`

Logic:
- Reads the raw demo CSV, parses dates, computes age from test/birth dates (warns on large discrepancies),
  and converts sex to numeric codes.
- Builds a clean table with `id`, `subid`, `age`, `sex`, `group`, and selected demographic fields.
- Merges rsfMRI QC by `subid` using `rest_fd_summary.csv`:
  - Keeps only `valid_subject == 1` in QC.
  - Inner-joins to demo, adds `meanFD`.
  - Filters `age < 26`, `group == ''`, and `meanFD` not null.
- Outputs processed demo CSV and a merged demo+QC CSV.
