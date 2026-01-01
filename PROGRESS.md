# PROGRESS.md

Real-time development log for the refactor of `data_driven_EF`.

## Current focus

## Completed

- 更新路径与文档为单一 EFNY 数据集结构（去除 data/ 与 outputs/ 的数据集层级）。
- 合并 EFNY 配置与文档到统一位置（`configs/paths.yaml` 与 `docs/workflow.md`）。
- 在 `docs/data_dictionary.md` 中补充 QC 与 FC 相关文件说明。
- 在 `docs/workflow.md` 新增预处理流程章节，并在 `docs/methods.md` 补充方法学细节。
- 重新整理 `docs/workflow.md` 与 `docs/methods.md`，修正中文编码乱码。
- 补充影像与行为预处理的详细步骤（xcp-d、头动 QC 标准、行为清洗与映射）。
- 重构 `src/` 目录：影像预处理与行为预处理分离，并统一命名风格。
- 将 `preprocess_efny_demo.py` 归类到行为数据预处理模块。
- 移除基于 task 配置表的旧指标流程，仅保留 `DEFAULT_TASK_CONFIG` 方案。
- 将行为指标计算与下游使用指标的配置迁移到 `configs/behavioral_metrics.yaml`。



## In progress

## Next up

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
