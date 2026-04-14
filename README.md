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

## Output locations (convention)

Common outputs:
