# EFNY dataset (current focus)

This document consolidates EFNY dataset-specific documentation previously split across multiple EFNY-specific documents under `docs/datasets/`.

Scope: EFNY-specific file conventions, preprocessing assumptions, and the intended run order of EFNY-related scripts. This document does not report empirical results.

## 1) EFNY data location and conventions (dataset-specific)

- Canonical data roots: `data/raw/EFNY/`, `data/interim/EFNY/`, `data/processed/EFNY/`, `outputs/EFNY/`, and `logs/EFNY/`.
- Runtime artifacts: `data/`, `outputs/`, and `logs/` are not version-controlled; paths below are expected locations used by scripts.
- Subject identifiers: EFNY scripts commonly use `subid`/`subject_id`/`subject_code` depending on the source table; verify the exact column names in the script you are running.

## 2) Behavioral data and EF metrics (dataset-specific)

### 2.1 Inputs and outputs

- Input directory (raw): `data/raw/EFNY/behavior_data/cibr_app_data/`
- Input format: per-subject Excel workbooks (`*.xlsx`)
- Output wide table (metrics): `data/processed/EFNY/table/metrics/EFNY_beh_metrics.csv`

Within the behavioral metric pipeline, `subject_code` is derived from the filename by removing the suffix `_GameData.xlsx` (and `file_name` stores the original basename).

### 2.2 Column normalization (xlsx → standardized DataFrame)

When reading each sheet, EFNY behavioral scripts normalize common column headers to English field names (see `src/metric_compute/efny/io.py`):

- `任务` → `task`
- `游戏序号` → `trial_index`
- `被试编号（用户账号）` → `subject_id`
- `被试姓名（真实姓名)` → `subject_name`
- `正式阶段刺激图片/Item名` → `item`
- `正式阶段正确答案` → `answer`
- `正式阶段被试按键` → `key`
- `绝对时间(待定)` → `abs_time`
- `相对时间(秒)` → `rt`

### 2.3 Task name normalization (sheet name → internal task key)

EFNY behavioral scripts map Excel sheet names to internal task keys (see `src/metric_compute/efny/main.py`), including:

- `*1back*` → `oneback_number` / `oneback_spatial` / `oneback_emotion` (by sheet name content)
- `*2back*` → `twoback_number` / `twoback_spatial` / `twoback_emotion`
- Other tasks typically keep their normalized sheet name (e.g., `FLANKER`, `SST`, `DT`, `EmotionSwitch`)

### 2.4 Trial-level preprocessing and QC assumptions

Most tasks using reaction time (RT) share a common preprocessing step (see `src/metric_compute/efny/preprocess.py`):

- Correctness: if `correct_trial` is absent, compute it from `answer == key` (nullable boolean).
- RT parsing: convert `rt` to numeric (`errors='coerce'`).
- Optional RT filtering: enforce task-specific `rt_min`/`rt_max`, then apply ±3 SD trimming on the retained RTs.
- Minimum valid-trial proportion: if too many trials are missing/filtered (relative to `min_prop`), the task is marked invalid and outputs `NaN` for that task’s metrics.

### 2.5 Output naming convention

Per-subject, per-task metrics are written as columns with the pattern:

- `{task_name}_{metric_name}` (e.g., `FLANKER_ACC`, `oneback_number_dprime`)

### 2.6 Metric families (high-level)

The EFNY behavioral pipeline produces a wide set of task-derived metrics (see `src/metric_compute/efny/main.py` and `src/metric_compute/efny/metrics.py`). Key families include:

- N-back: `ACC`, correct-trial RT summary, hit/false-alarm rates, `d'` (with small-sample corrections).
- Conflict tasks (Flanker / ColorStroop / EmotionStroop): condition-specific accuracy/RT and contrasts (incongruent − congruent).
- Switching tasks (DCCS / DT / EmotionSwitch): repeat vs. switch metrics and switch costs.
- SST: go-trial accuracy/RT, `SSRT` (integration method), `Mean_SSD`, stop accuracy.
- Go/NoGo tasks (e.g., GNG, CPT): go/no-go accuracy, RT summaries, `d'`.
- Additional tasks (e.g., ZYST, FZSS, KT): task-specific summaries as implemented in `src/metric_compute/efny/metrics.py`.

For exact definitions (e.g., how conditions are parsed from `item`, how “no-press” is represented), treat the code as authoritative.

## 3) Neuroimaging preprocessing and QC (dataset-specific)

This section summarizes the intended run order for EFNY rs-fMRI preprocessing scripts under `src/preprocess/`.

### 3.1 Intended run order

1) xcp-d (rest preprocessing)
2) head motion screening (QC summary)
3) demographic preprocessing and QC merge

### 3.2 xcp-d rest preprocessing

Scripts:

- `src/preprocess/batch_run_xcpd.sh`
- `src/preprocess/xcpd_36p.sh`

Operational summary:

- `batch_run_xcpd.sh` reads subject IDs from `data/processed/EFNY/table/sublist/mri_sublist.txt` and submits one job per subject.
- Each job runs xcp-d on rs-fMRI with 36P nuisance regression and band-pass filtering; outputs are written under `data/interim/EFNY/MRI_data/xcpd_rest` (see script for exact version/flags).

### 3.3 Head motion QC summary

Script:

- `src/preprocess/screen_head_motion_efny.py`

Operational summary:

- Computes per-run frame count, mean framewise displacement (FD), and the proportion of frames with FD > 0.3.
- Defines run- and subject-level validity flags and writes a QC summary table `data/interim/EFNY/table/qc/rest_fd_summary.csv` (see script for the precise criteria).

### 3.4 Demographics preprocessing and QC merge

Script:

- `src/preprocess/preprocess_efny_demo.py`

Operational summary:

- Parses and cleans EFNY demographics, computes age from dates, standardizes sex coding, and merges the rs-fMRI QC summary by subject ID.
- Produces processed demographic tables, including a demo+QC merged table (see script for exact output paths and filtering rules).

## 4) EFNY in the general modeling workflow (general + EFNY assumptions)

This section distinguishes general workflow requirements (dataset-agnostic) from EFNY-specific alignment assumptions.

### 4.1 General workflow requirements (dataset-agnostic)

- Evaluation must use nested cross-validation; all preprocessing must be fit on training splits only and applied to held-out data.
- Real vs. permutation runs are executed via a single entrypoint: `src/scripts/run_single_task.py` (real: `task_id=0`; permutations: `task_id>=1`).
- Supported adaptive model types (as documented by the repository): `adaptive_pls`, `adaptive_scca`, `adaptive_rcca`.

### 4.2 EFNY-specific alignment assumptions (verify before running)

Some modeling components assume that brain features, behavioral features, and subject lists are already aligned row-wise (i.e., they check shape consistency but do not necessarily reindex/merge by subject ID). Before any modeling run, confirm that:

- The subject list used to generate/connectome features matches the subject list used to filter behavioral tables.
- Behavioral tables have not been reordered relative to the subject list without an explicit merge/reindex step.

If alignment is not guaranteed, treat the run as invalid until the inputs are explicitly aligned.

### 4.3 Atlas selection for FC features (Schaefer)

The FC feature path encodes the atlas choice. Supported resolutions are Schaefer100, Schaefer200, and Schaefer400. To switch atlas resolution, update the `brain_file` entry in `configs/datasets/EFNY.yaml` by replacing the atlas name in both the directory and filename.

Path pattern:

- `fc_vector/<Atlas>/EFNY_<Atlas>_FC_matrix.npy`

Example substitutions:

- `fc_vector/Schaefer100/EFNY_Schaefer100_FC_matrix.npy`
- `fc_vector/Schaefer200/EFNY_Schaefer200_FC_matrix.npy`
- `fc_vector/Schaefer400/EFNY_Schaefer400_FC_matrix.npy`

## 5) Updating this document

When EFNY preprocessing or file conventions change, update this document and add a short session note under `docs/sessions/` describing:

- what changed,
- why it changed,
- which scripts/paths are affected.
