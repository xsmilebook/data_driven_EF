# 数据字典（建设中）

本文件提供关键表与常用字段的轻量级索引。EFNY 的数据集假设与模式见 `docs/workflow.md` 的 EFNY 说明部分。

## EFNY（当前重点）

权威数据集说明：

- `docs/workflow.md`（EFNY 数据集说明）

常用表（路径为约定，见 `configs/paths.yaml`）：

- `data/processed/table/demo/EFNY_demo_processed.csv`: 清洗后的基本人口学信息。
- `data/processed/table/demo/EFNY_demo_with_rsfmri.csv`: 人口学信息与 rs-fMRI 质控合并（如包含 `meanFD` 等运动摘要字段）。
- `data/processed/table/metrics/EFNY_beh_metrics.csv`: 按被试汇总的行为指标宽表。
- `data/processed/table/demo/EFNY_behavioral_data.csv`: 人口学 + 行为指标合并表（如有生成）。
- `data/processed/table/sublist/sublist.txt`: 分析用被试列表（如有生成）。

## 约定

- 标识列因数据源而异；跨表合并时优先使用明确的 `subid` 列。
- 行为数值指标应以数值列存储；缺失值需统一表示（例如空值/NA 解析为 NaN）。
