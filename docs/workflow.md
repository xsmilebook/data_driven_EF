# workflow

## 预处理脚本组织原则

本项目优先采用浅层目录与显式脚本命名，避免过度抽象，便于科研工作者直接定位、修改和重跑单一步骤。

## 影像数据

- `src/imaging/fmriprep/`：仅负责 `fmriprep` 任务提交、产物检查和 QC。
- `src/imaging/xcpd/`：仅负责 `xcpd` 任务提交、产物检查和 QC。
- `src/imaging/connectivity/`：负责功能连接等影像指标提取与后处理。

影像流程拆分为独立子目录，是因为 `fmriprep` 和 `xcpd` 运行时间长、通常需要并行提交，失败后应能局部重跑，而不应绑定在单个长脚本中。

## 行为数据

- `src/behavior/app/`：处理 app 行为任务。
- `src/behavior/task_fmri/`：处理 task fMRI 的行为日志，如 Psychopy 导出。
- `src/behavior/inventory/`：处理行为量表。
- `src/behavior/demography/`：处理人口学信息。

其中，`app` 与 `task_fmri` 均遵循三阶段组织：

1. `check_format.py`：检查原始文件格式、列名和关键标识字段。
2. `clean.py`：执行试次级与被试级清洗。
3. `metrics.py`：计算行为指标。

量表模块采用相同思路，但第三阶段命名为 `score.py`，用于计分与因子计算。

## task 特异规则

对于 app 行为任务和 task fMRI 行为任务，task 特异规则下沉到各自目录下的 `tasks/` 子目录，不在主流程脚本中堆积大量条件分支。

- `check_format.py`、`clean.py`、`metrics.py` 负责通用流程框架。
- `tasks/` 中的文件负责各任务的特异列映射、清洗规则和指标定义。

## scripts 目录

`scripts/` 仅放执行入口，不放核心逻辑。

- `scripts/imaging/`：提交或汇总 `fmriprep`、`xcpd`、connectivity 相关任务。
- `scripts/behavior/`：运行行为、量表与人口学的 `check_format`、`clean`、`metrics` 或 `score` 步骤，并在最后构建分析表。

这种安排的目标是：

- 保持每个脚本职责单一，便于局部重跑。
- 让用户通过脚本名即可判断其输入、输出和执行阶段。
- 在不增加过多抽象层的前提下，为多 task 和长耗时影像流程提供清晰边界。

## app_data 清洗参数探索

对于 `app_data` 的探索性试次清洗参数分析，当前在 [notebooks/app_data_eda/app_data_trial_cleaning_parameter_exploration.ipynb](/D:/projects/data_driven_EF/notebooks/app_data_eda/app_data_trial_cleaning_parameter_exploration.ipynb) 中维护独立 notebook。

该 notebook 直接读取 `data/raw/*/app_data/*.xlsx`，围绕 `相对时间(秒)` 提供以下统计：

- 当 `相对时间(秒) < 0.1` 或 `< 0.2` 时，平均每个被试有多少个 sheet 出现无效试次，以及平均每个 sheet 有多少个无效试次。
- 当每个 sheet 内 `相对时间(秒) > mean + 3 s.d.` 时，平均每个 sheet 有多少个高阈值无效试次。

该 notebook 的定位是参数探索，不替代正式预处理脚本。
