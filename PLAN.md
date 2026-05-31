# Plan

## THU app 行为数据预处理流水线

目标：实现 `data/raw/THU/app_data/*.xlsx` 的正式三阶段预处理入口，输出格式 QC、
清洗试次长表、任务 QC、指标长表和指标宽表。

执行项：

- 新增 `configs/paths.yaml` 与 `configs/behavioral_metrics.yaml`。
- 新增 `src/common.py` 和 `src/behavior/app/` 下的格式检查、清洗、指标计算与任务规则模块。
- 新增 `scripts/behavior/app_check_format.py`、`app_clean.py`、`app_metrics.py` 三个入口。
- 仅按 `[0.2, 10]` 秒闭区间筛选 RT，不执行被试任务内 `mean + 3 SD` 修剪。
- 仅使用 RT 合格试次计算准确率与 RT 指标。
- 按任务随机正确率阈值标记任务 QC；KT 与 ZYST 不设置阈值。
- 对未确认的 EmotionStroop 条件映射保留显式配置接口，不猜测条件方向。
- 使用 3 个 workbook 做非持久化冒烟测试，清理临时结果后提交代码与文档。

约束：

- v1 仅实现 THU `app_data`，不扩展 XY、BNU、inventory、demography 或 task-fMRI。
- 不运行 753 个 workbook 的全量任务，仅提供本地和单节点 sbatch 提交命令。
