# EFNY 数据集（当前重点）

本文件整合 EFNY 数据集的专用说明，包括文件命名约定、预处理假设与相关脚本的预期运行顺序。本文件不报告任何经验性结果。

范围：仅限 EFNY 特定的文件约定、预处理假设与脚本运行顺序。实现细节以代码为准。

## 1) EFNY 数据位置与约定（数据集特定）

- 规范数据根目录：`data/raw/EFNY/`、`data/interim/EFNY/`、`data/processed/EFNY/`、`outputs/EFNY/`、`outputs/EFNY/logs/`。
- 运行时产物：`data/` 与 `outputs/`（含 `outputs/<DATASET>/logs/`）不纳入版本控制；以下路径为脚本默认/预期位置。
- 被试标识：EFNY 脚本中常见字段有 `subid`/`subject_id`/`subject_code`；以具体脚本为准。

## 2) 行为数据与 EF 指标（数据集特定）

### 2.1 输入与输出

- 原始输入目录：`data/raw/EFNY/behavior_data/cibr_app_data/`
- 输入格式：每位被试一个 Excel 工作簿（`*.xlsx`）
- 输出行为指标宽表：`data/processed/EFNY/table/metrics/EFNY_beh_metrics.csv`

在行为指标流水线中，`subject_code` 由文件名去掉后缀 `_GameData.xlsx` 得到，原始文件名保存在 `file_name`。

### 2.2 列名标准化（xlsx -> 标准化 DataFrame）

脚本会将常见中文列名统一到英文字段（见 `src/metric_compute/efny/io.py`）。请以源码中的映射表为准。

### 2.3 任务名称标准化（sheet 名称 -> 内部任务键）

Excel 表单名称会映射到内部任务键（见 `src/metric_compute/efny/main.py`），包括：

- `*1back*` -> `oneback_number` / `oneback_spatial` / `oneback_emotion`（依据 sheet 名称包含的关键字）
- `*2back*` -> `twoback_number` / `twoback_spatial` / `twoback_emotion`
- 其他任务通常保留规范化后的 sheet 名称（如 `FLANKER`, `SST`, `DT`, `EmotionSwitch`）

### 2.4 试次级预处理与 QC 假设

对依赖 RT 的任务，常见的预处理步骤见 `src/metric_compute/efny/preprocess.py`：

- 正确性：若缺少 `correct_trial`，则以 `answer == key` 计算。
- RT 解析：将 `rt` 转为数值（`errors='coerce'`）。
- 可选 RT 过滤：先按任务特定 `rt_min`/`rt_max` 过滤，再进行 ±3 SD 修剪。
- 最低有效试次比例：若有效试次比例低于阈值（`min_prop`），该任务输出记为 `NaN`。

### 2.5 输出命名约定

按被试、按任务的指标以列形式输出：

- `{task_name}_{metric_name}`（例如 `FLANKER_ACC`, `oneback_number_dprime`）

### 2.6 指标类型概览（高层）

行为流水线输出多类任务指标（见 `src/metric_compute/efny/main.py` 与 `src/metric_compute/efny/metrics.py`），例如：

- N-back：`ACC`、RT 汇总、命中/虚警率、`d'`（含小样本修正）。
- 冲突任务（Flanker / ColorStroop / EmotionStroop）：条件准确率/RT 与差值指标。
- 切换任务（DCCS / DT / EmotionSwitch）：重复/切换指标与切换成本。
- SST：go 试次准确率/RT、`SSRT`（积分法）、`Mean_SSD`、stop 准确率。
- Go/NoGo（如 GNG, CPT）：go/no-go 准确率、RT 汇总、`d'`。
- 其他任务（如 ZYST, FZSS, KT）：见实现细节。

若需精确定义（如条件解析规则、无按键的表示方式），以源码为准。

## 3) 神经影像预处理与 QC（数据集特定）

本节概述 `src/preprocess/` 下 EFNY rs-fMRI 预处理脚本的预期顺序。

### 3.1 预期运行顺序

1) xcp-d（rest 预处理）
2) 头动筛查（QC 汇总）
3) 人口学预处理与 QC 合并

### 3.2 xcp-d 静息态预处理

脚本：

- `src/preprocess/batch_run_xcpd.sh`
- `src/preprocess/xcpd_36p.sh`

概述：

- `batch_run_xcpd.sh` 从 `data/processed/EFNY/table/sublist/mri_sublist.txt` 读取被试列表，每个被试提交一个作业。
- 每个作业对 rs-fMRI 执行 36P 回归与带通滤波；输出位于 `data/interim/EFNY/MRI_data/xcpd_rest`（具体参数以脚本为准）。

### 3.3 头动 QC 汇总

脚本：

- `src/preprocess/screen_head_motion_efny.py`

概述：

- 计算每 run 的帧数、平均 FD、FD > 0.3 的帧比例。
- 生成 run 级与被试级有效性标记，并输出 QC 汇总表 `data/interim/EFNY/table/qc/rest_fd_summary.csv`（具体阈值以脚本为准）。

### 3.4 人口学预处理与 QC 合并

脚本：

- `src/preprocess/preprocess_efny_demo.py`

概述：

- 清洗人口学字段，计算年龄，统一性别编码，并按被试 ID 合并 rs-fMRI QC 汇总表。
- 生成处理后的人口学表与 demo+QC 合并表（具体输出路径与规则以脚本为准）。

## 4) EFNY 在建模流程中的位置（通用 + EFNY 假设）

本节区分通用流程要求与 EFNY 特定的对齐假设。

### 4.1 通用流程要求（数据集无关）

- 评估必须使用嵌套交叉验证；所有预处理仅在训练折拟合并应用到留出数据。
- 真实与置换分析使用同一入口：`scripts/run_single_task.py`（真实：`task_id=0`；置换：`task_id>=1`）。
- 当前支持的自适应模型类型：`adaptive_pls`、`adaptive_scca`、`adaptive_rcca`。

### 4.2 EFNY 特定对齐假设（运行前必须核对）

部分建模组件仅检查形状一致性，未必执行基于 ID 的重排/合并。运行前应确认：

- 脑特征、行为特征、被试列表来源一致且按同一顺序对齐。
- 若存在重排或过滤，必须通过明确的 ID 合并/重排确保对齐。

若对齐无法保证，结果应视为无效。

### 4.3 FC 特征的 Atlas 选择（Schaefer）

FC 特征路径包含 atlas 名称。支持 Schaefer100/200/400。切换时需修改 `configs/datasets/EFNY.yaml` 中 `brain_file` 的目录与文件名。

路径模式：

- `fc_vector/<Atlas>/EFNY_<Atlas>_FC_matrix.npy`

示例：

- `fc_vector/Schaefer100/EFNY_Schaefer100_FC_matrix.npy`
- `fc_vector/Schaefer200/EFNY_Schaefer200_FC_matrix.npy`
- `fc_vector/Schaefer400/EFNY_Schaefer400_FC_matrix.npy`

## 5) 更新本文件

当 EFNY 预处理或文件约定变更时，请更新本文件，并在 `docs/sessions/` 下添加一条简短会话记录，说明：

- 变更内容
- 变更原因
- 受影响的脚本/路径
