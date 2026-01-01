# data_driven_EF

本仓库包含 EF（执行功能）研究的端到端流程：
- 数据 QC / 被试列表
- 功能连接（FC）计算与向量化
- 行为指标计算
- 脑-行为关联分析（仅保留自适应模型：`adaptive_pls` / `adaptive_scca` / `adaptive_rcca`）

下面仅列出常用脚本的执行方式与关键参数。

## 目录结构（核心）

```
src/
  imaging_preprocess/    # 影像预处理与功能连接
  behavioral_preprocess/ # 行为数据预处理与指标计算
  models/                # 脑-行为关联（建模评估/嵌套 CV）
  scripts/               # 入口脚本 + HPC 提交脚本
  result_summary/        # 结果汇总脚本
```

## 输出位置（约定）

常用输出：
- QC / 中间结果：`data/interim/...`
- 表格与处理后产物：`data/processed/...`
- 脑-行为关联结果：`outputs/results/...`（可通过 `--output_dir` 指定）
- 大型数组（如每折 `X_scores/Y_scores`）会保存到同目录下 `artifacts/`，JSON/NPZ 仅保留索引与路径
