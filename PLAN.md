# Plan

## 行为数据预处理方法文档完善

目标：完善 `docs/methods.md` 中 `behavioral data preprocess` 部分，将 THU
`app_data` 行为任务预处理写成可执行的 v1 方法规格；本次仅修改文档，不实现代码、
不生成 `data/` 或 `outputs/` 下运行产物。

执行项：

- 修正行为预处理标题拼写。
- 明确 v1 输入范围为 `data/raw/THU/app_data/*.xlsx`，每个 workbook 为 1 名被试，
  每个 sheet 为 1 个任务表。
- 记录 18 类 app 任务 sheet 的支持范围。
- 规范中文原始列到英文标准字段的映射。
- 区分 `valid_for_acc` 与 `valid_for_rt`，避免将允许无反应的试次误排除出准确率分母。
- 写明 RT 清洗默认参数：`rt_min=0.2`、`rt_max=10`、`rt_outlier_sd=3`、
  `min_valid_prop=0.5`。
- 分任务写明 N-back、冲突任务、任务切换、SST、GNG/CPT、ZYST、FZSS、KT 的条件定义与指标输出。
- 补充清洗长表、QC 表、指标长表、指标宽表的字段约定。
- 补充非持久化冒烟验证与 `git diff --check` 检查要求。

约束：

- 不创建 `src/behavior`、`scripts/behavior` 或 `configs/behavioral_metrics.yaml`。
- 不修改 `data/` 或 `outputs/`。
- 会话日志按当前日期写入 `docs/sessions/2026-05-30.md`。
