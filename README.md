# data_driven_EF
This repository contains end-to-end pipeline for EF analysis:
- neuroimaging and behavioral data preprocessing
- functional connectivity (FC) calculation and vectorization
- behavioral metrics calculation
- brain-behavior association analysis (only adaptive models: `adaptive_pls` / `adaptive_scca` / `adaptive_rcca`)

## Directory structure (core)

```
src/
  imaging_preprocess/    # 影像预处理与功能连接
  behavioral_preprocess/ # 行为数据预处理与指标计算
  models/                # 脑-行为关联（建模评估/嵌套 CV）
  scripts/               # 入口脚本 + HPC 提交脚本
  result_summary/        # 结果汇总脚本
```

## Output locations (convention)

Common outputs：

