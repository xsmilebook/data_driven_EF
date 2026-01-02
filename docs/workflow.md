# 工作流（工程）

本文档描述在 `ARCHITECTURE.md` 固定目录结构下，脚本入口的标准化调用方式。

## 标准 CLI 形式

`scripts/` 下所有可运行入口应接受：

- `--dataset <DATASET_NAME>`
- `--config <PATH_TO_CONFIGS_PATHS_YAML>`

示例：

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

EFNY 相关假设配置在 `configs/paths.yaml` 的 `dataset` 段落中。数据集无关的分析默认值可配置在 `configs/analysis.yaml`。

## data/ 与 outputs/ 约定

- `data/raw/`: 原始输入（不由流水线脚本生成）
- `data/interim/`: 中间产物（例如 MRI 预处理或连接组中间结果）
- `data/processed/`: 可复用的处理后数据（如表格、FC 向量特征）
- `outputs/`: 运行产物（结果、图表）
- `outputs/logs/`: 运行日志（脚本日志、SLURM stdout/stderr）

部分外部输入（如 fMRIPrep 输出）可能位于仓库外；请在 `configs/paths.yaml` 的 `dataset.external_inputs` 下配置绝对路径。

## 预处理流程（影像与行为）

本节描述推荐的预处理顺序与关键产物，细节以脚本为准。

### 影像预处理与功能连接

1) rs-fMRI 预处理（xcp-d）。
   - 脚本：`src/imaging_preprocess/batch_run_xcpd.sh`、`src/imaging_preprocess/xcpd_36p.sh`
   - 输入：`data/raw/MRI_data/`
   - 输出：`data/interim/MRI_data/xcpd_rest`
2) 头动 QC 汇总。
   - 脚本：`src/imaging_preprocess/screen_head_motion_efny.py`
   - 输出：`data/interim/table/qc/rest_fd_summary.csv`
   - 有效 run 判定：`frame == 180`、`high_ratio <= 0.25` 且 `mean_fd <= upper_limit`。
   - `high_ratio` 为 `framewise_displacement > 0.3` 的比例；`upper_limit` 来自全体 FD 值的 `Q3 + 1.5*IQR`。
   - `valid_subject` 标准：有效 run 数 `valid_num >= 2`。
3) 单被试 FC 与 Fisher-Z。
   - 脚本：`src/imaging_preprocess/compute_fc_schaefer.py`、`src/imaging_preprocess/fisher_z_fc.py`
   - 输出：`data/interim/functional_conn/rest/` 与 `data/interim/functional_conn_z/rest/`
4) FC 向量化特征。
   - 脚本：`src/imaging_preprocess/convert_fc_vector.py`
   - 输出：`data/processed/fc_vector/`

### 行为数据预处理与指标计算

1) app 行为数据规范化（如需）。
   - 脚本：`src/behavioral_preprocess/app_data/format_app_data.py`
   - 输入：`data/raw/behavior_data/cibr_app_data/`
2) 行为指标计算。
   - 脚本：`src/behavioral_preprocess/metrics/compute_efny_metrics.py`
   - 输出：`data/processed/table/metrics/EFNY_beh_metrics.csv`
   - 列名规范化：见 `src/behavioral_preprocess/metrics/efny/io.py` 的 `normalize_columns`。
   - 任务映射：见 `src/behavioral_preprocess/metrics/efny/main.py`，按 sheet 名称映射到内部任务键。
   - 试次清洗：见 `src/behavioral_preprocess/metrics/efny/preprocess.py` 的 `prepare_trials`。
3) 人口学与指标合并。
   - 脚本：`src/behavioral_preprocess/app_data/build_behavioral_data.py`
   - 输出：`data/processed/table/demo/EFNY_behavioral_data.csv`

## 结果目录

`src/result_summary/` 下的汇总脚本默认使用：

- `outputs/results`

如有需要，可通过 `--results_root` 覆盖。

示例：

```bash
python -m src.result_summary.summarize_real_perm_scores --dataset EFNY --config configs/paths.yaml --analysis_type both --atlas <atlas> --model_type <model>
```

## 快速自检（dry-run）

在仓库根目录执行：

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

该命令验证导入与配置解析，不会读取 `data/` 或写入 `outputs/`。

## 集群执行（SLURM）

提交脚本：

- `scripts/submit_hpc_real.sh`
- `scripts/submit_hpc_perm.sh`

这些脚本将 `#SBATCH --chdir` 设为集群项目根目录，并将 SLURM stdout/stderr 与任务日志写入 `outputs/logs/...`。
注意：SLURM 的 `#SBATCH --output/--error` 路径为静态字符串，无法展开环境变量，因此在脚本头部保持固定路径。

示例：

```bash
sbatch scripts/submit_hpc_real.sh
```

## EFNY 数据集说明（唯一数据集）

本节整合 EFNY 的数据约定、预处理假设与相关脚本顺序，避免分散在单独的数据集文档中。本节不报告任何结果，细节以源码为准。

### 1) 数据位置与约定

- 规范根目录：`data/raw/`、`data/interim/`、`data/processed/`、`outputs/`、`outputs/logs/`。
- 运行时产物：`data/` 与 `outputs/`（含 `outputs/logs/`）不纳入版本控制。
- 被试标识：常见字段为 `subid`/`subject_id`/`subject_code`，以具体脚本为准。

### 2) 行为数据与 EF 指标

输入与输出：

- 原始输入目录：`data/raw/behavior_data/cibr_app_data/`
- 输入格式：每位被试一个 Excel 工作簿（`*.xlsx`）
- 输出指标宽表：`data/processed/table/metrics/EFNY_beh_metrics.csv`

列名与任务规范：

- 列名统一与任务映射见 `src/behavioral_preprocess/metrics/efny/io.py` 与 `src/behavioral_preprocess/metrics/efny/main.py`。
- 任务名映射遵循：`*1back*` -> `oneback_*`，`*2back*` -> `twoback_*`，其余保留规范化名称。

试次级预处理（见 `src/behavioral_preprocess/metrics/efny/preprocess.py`）：

- 缺失 `correct_trial` 时用 `answer == key` 计算。
- `rt` 转数值（`errors='coerce'`），可选按 `rt_min`/`rt_max` 过滤并进行 ±3 SD 修剪。
- 有效试次比例低于阈值（`min_prop`）时输出 `NaN`。

输出列名模式：

- `{task_name}_{metric_name}`（如 `FLANKER_ACC`, `oneback_number_dprime`）。

### 行为数据探索性分析（单被试试次统计）

目的：对单个 Excel 工作簿（单被试）按指标计算口径统计试次数量与条件分布，用于建模可行性评估与数据质量自检。

默认输入文件由 `configs/paths.yaml` 的 `dataset.behavioral.reference_game_data_file` 指定，位于 `dataset.behavioral.app_data_dir` 下。

运行：

```bash
python -m scripts.eda_behavior_trials --dataset EFNY --config configs/paths.yaml
```

输出：

- 报告写入 `configs/paths.yaml` 的 `docs_reports_root`（默认 `docs/reports/`），文件名根据工作簿自动生成。
- 可通过 `--excel-file`（绝对路径或仓库相对路径）与 `--report-out` 覆盖默认输入/输出位置。

### 3) 神经影像预处理与 QC

预期顺序：

1) xcp-d（rest 预处理）
2) 头动筛查（QC 汇总）
3) 人口学预处理与 QC 合并

xcp-d：

- 脚本：`src/imaging_preprocess/batch_run_xcpd.sh`、`src/imaging_preprocess/xcpd_36p.sh`
- 被试列表：`data/processed/table/sublist/mri_sublist.txt`
- 输出目录：`data/interim/MRI_data/xcpd_rest`

头动 QC：

- 脚本：`src/imaging_preprocess/screen_head_motion_efny.py`
- 输出：`data/interim/table/qc/rest_fd_summary.csv`

人口学处理：

- 脚本：`src/behavioral_preprocess/app_data/preprocess_efny_demo.py`
- 输出：人口学清洗表与 demo+QC 合并表（以脚本为准）

### 4) 建模流程中的 EFNY 假设

- 评估使用嵌套交叉验证；所有预处理仅在训练折拟合并应用到留出数据。
- 真实与置换分析共用入口：`scripts/run_single_task.py`（真实 `task_id=0`，置换 `task_id>=1`）。
- 当前支持的模型：`adaptive_pls`、`adaptive_scca`、`adaptive_rcca`。

对齐要求：

- 脑特征、行为特征与被试列表必须基于 ID 明确对齐。
- 若存在重排/过滤，应显式合并与重排；仅靠行序一致性视为不可靠。

Atlas 选择（Schaefer）：

- 通过 `configs/paths.yaml` 的 `dataset.files.brain_file` 切换 Schaefer100/200/400。
- 路径模式：`fc_vector/<Atlas>/EFNY_<Atlas>_FC_matrix.npy`。

### 5) 更新说明

当预处理或文件约定变更时，请在 `docs/sessions/` 记录变更内容、原因与涉及脚本/路径。

