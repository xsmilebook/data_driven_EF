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
- 当 `相对时间(秒) < 0.1` 或 `< 0.2` 时，在这些无效试次内，基于 `正式阶段正确答案 == 正式阶段被试按键` 计算平均正确率。
- 当每个 sheet 内 `相对时间(秒) > mean + 3 s.d.` 时，平均每个 sheet 有多少个高阈值无效试次。

该 notebook 的定位是参数探索，不替代正式预处理脚本。

## THU app_data 正式预处理

THU `app_data` 的正式流水线已实现为三个顺序执行入口：

```powershell
uv run python -m scripts.behavior.app_check_format --dataset THU
uv run python -m scripts.behavior.app_clean --dataset THU
uv run python -m scripts.behavior.app_metrics --dataset THU
```

默认输入为 `data/raw/THU/app_data/*.xlsx`，默认输出目录为
`data/processed/THU/behavioral_metrics/`。路径在 `configs/paths.yaml` 中配置，
字段映射、RT 阈值、任务注册和随机正确率阈值在
`configs/behavioral_metrics.yaml` 中配置。枚举 workbook 时自动忽略 Excel 打开文件
产生的 `~$*.xlsx` 临时锁文件。

三个入口分别生成：

- `app_format_qc.csv`：workbook/sheet 格式检查记录。
- `app_trials_clean.csv`：标准化试次长表及 RT 排除原因。
- `app_task_qc.csv`：任务级准确率阈值检查记录。
- `app_metrics_long.csv`：长格式任务指标。
- `app_metrics_wide.csv`：用于下游分析的被试级宽表。

小样本验证可使用：

```powershell
uv run python -m scripts.behavior.app_check_format --dataset THU --limit 3 --output-dir temp/smoke_app_behavior
uv run python -m scripts.behavior.app_clean --dataset THU --limit 3 --output-dir temp/smoke_app_behavior
uv run python -m scripts.behavior.app_metrics --dataset THU --output-dir temp/smoke_app_behavior
```

全量运行预计适合单节点顺序执行。SLURM 示例：

```bash
#!/bin/bash
#SBATCH --job-name=ef_app_behavior
#SBATCH --output=outputs/logs/ef_app_behavior_%j.out
#SBATCH --error=outputs/logs/ef_app_behavior_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

cd /path/to/data_driven_EF
uv run python -m scripts.behavior.app_check_format --dataset THU
uv run python -m scripts.behavior.app_clean --dataset THU
uv run python -m scripts.behavior.app_metrics --dataset THU
```

正式清洗将准确率有效试次与 RT 有效试次分开记录。RT 为空的 trial 仍进入准确率
分母，仅排除 RT 指标；RT 非空但超出 `[0.2, 10]` 秒闭区间时同时排除准确率和 RT
指标。任务 QC 表同时写入 `n_trials_acc_valid` 与 `n_trials_rt_valid`。

EmotionStroop 的一致/不一致条件映射已写入 `configs/behavioral_metrics.yaml`：
`e4`、`e8`、`e12`、`e16`、`e20`、`e24`、`e28`、`e32` 为一致条件，其余
`e1-e32` 编码为不一致条件。

Emotion2Back 中 item 全空的 sheet 记录 `nback_item_missing`。由于 ACC 可直接由
`answer == key` 计算，这类 sheet 在长表和宽表中继续保留 ACC；依赖 item 的
`Hit_Rate`、`FA_Rate` 和 `dprime` 保持为空。若 ACC 同时低于随机阈值，任务 QC
仍标记失败。

## THU app_data 指标探索报告

处理后指标的描述性 EDA 在
[notebooks/app_data_eda/app_metrics_eda_report.ipynb](/D:/projects/data_driven_EF/notebooks/app_data_eda/app_metrics_eda_report.ipynb)
中维护。该 notebook 不重新读取原始 Excel，而是使用
`data/processed/THU/behavioral_metrics/` 下的正式产物：

- `app_trials_clean.csv`：分块读取，仅汇总 trial 排除原因。
- `app_format_qc.csv` 与 `app_task_qc.csv`：汇总格式检查、任务覆盖率和任务 QC。
- `app_metrics_long.csv` 与 `app_metrics_wide.csv`：分析指标缺失、分布、异常值候选和
  相关矩阵。

运行与导出命令：

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/app_data_eda/app_metrics_eda_report.ipynb --ExecutePreprocessor.timeout=900
uv run jupyter nbconvert --to html --no-input notebooks/app_data_eda/app_metrics_eda_report.ipynb --output-dir outputs/results/app_data_eda --output app_metrics_eda_report.html
```

产物写入以下 ignored 目录：

- `outputs/results/app_data_eda/app_metrics_eda_report.html`：隐藏代码输入的聚合 HTML 报告。
- `outputs/figures/app_data_eda/`：任务 QC、缺失、分布和 Spearman 相关热图。
- `outputs/tables/app_data_eda/`：描述统计、Pearson/Spearman 相关矩阵、pairwise N
  以及内部 QC 表。

HTML 仅展示聚合信息。文件名包含 `_internal` 的 CSV 可包含 `subject_code`，只用于
内部定位复核对象，不包含姓名或原始账号。IQR 异常值候选仅用于人工复核，不会自动
进入排除规则。

## THU app_data SSM/DDM 建模

SSM/DDM 阶段是行为预处理后的可选独立步骤，读取
`data/processed/THU/behavioral_metrics/app_trials_clean.csv`，不改变现有
`app_metrics_long.csv` 或 `app_metrics_wide.csv`。

小样本 pilot 示例：

```powershell
uv run python -m scripts.behavior.app_ssm --dataset THU --model dccs_overall_ddm --mode pilot --max-subjects 3
```

运行全部已配置模型：

```powershell
uv run python -m scripts.behavior.app_ssm --dataset THU --model all --mode pilot --max-subjects 3
```

输出写入 `data/processed/THU/behavioral_metrics/ssm/`：

- `*_trials.csv`：对应模型的入模 trial 表。
- `*_qc.csv`：入模 trial 数、被试数、响应水平和条件计数。
- `*_summary.csv`：posterior 参数摘要与诊断字段。
- `*_trace.nc`：`InferenceData` posterior trace。

当前 HSSM 依赖栈在 Python 3.12 下导入失败，因此项目环境已降级为 Python 3.11。
`.python-version`、`pyproject.toml` 与 `uv.lock` 均以 Python 3.11 为准，虚拟环境仍固定
为仓库根目录 `.venv`。
