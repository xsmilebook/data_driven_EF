# data_driven_EF

This repository contains end-to-end pipeline for EF analysis:
- neuroimaging and behavioral data preprocessing
- functional connectivity (FC) calculation and vectorization
- behavioral metrics calculation
- brain-behavior association analysis (only adaptive models: `adaptive_pls` / `adaptive_scca` / `adaptive_rcca`)

## Directory structure (core)

```text
src/
  common.py              # 薄通用工具
  imaging/               # 影像预处理与影像指标提取
  behavior/              # app/task-fMRI/量表/demography 预处理

scripts/
  imaging/               # fmriprep/xcpd/connectivity 入口脚本
  behavior/              # check_format/clean/metrics/score 入口脚本
```

Behavior preprocessing follows a fixed three-stage pattern where possible:
- `check_format`: check raw file format, column names, and required identifiers
- `clean`: apply trial-level and subject-level cleaning
- `metrics` / `score`: compute task metrics or scale scores

## THU app behavioral preprocessing

The implemented THU app pipeline runs in three ordered stages:

```powershell
uv run python -m scripts.behavior.app_check_format --dataset THU
uv run python -m scripts.behavior.app_clean --dataset THU
uv run python -m scripts.behavior.app_metrics --dataset THU
```

Default outputs are written to `data/processed/THU/behavioral_metrics/`.
See `docs/workflow.md` and `docs/methods.md` for execution details and metric definitions.

## THU app metrics EDA

The processed THU app metrics can be summarized with:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/app_data_eda/app_metrics_eda_report.ipynb --ExecutePreprocessor.timeout=900
uv run jupyter nbconvert --to html --no-input notebooks/app_data_eda/app_metrics_eda_report.ipynb --output-dir outputs/results/app_data_eda --output app_metrics_eda_report.html
```

The notebook reads the processed CSV files, writes aggregate figures and tables under
`outputs/`, and exports an HTML report without code inputs or subject-level rows. See
`docs/reports/app_metrics_eda.md` for the current descriptive summary.

## Output locations (convention)

Common outputs:
