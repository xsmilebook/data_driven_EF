# data_driven_EF

本仓库包含 EF（执行功能）研究的端到端流程：
- 数据 QC / 被试列表
- 功能连接（FC）计算与向量化
- 行为指标计算
- 脑-行为关联分析（仅保留自适应模型：`adaptive_pls` / `adaptive_scca` / `adaptive_rcca`）

下面只保留**最常用脚本的执行方式与关键参数**。

## 目录结构（核心）

```
src/
  preprocess/        # QC、表格与被试列表
  functional_conn/   # FC 计算、Z变换、向量化、可视化
  metric_compute/    # 行为指标计算与可视化
  models/            # 脑-行为关联（建模/评估/嵌套CV）
  scripts/           # 入口脚本 + HPC 提交脚本
  result_summary/    # 汇总脚本
```

## 快速开始（常用脚本）

### 1) 预处理 / QC / 被试列表（`src/preprocess`）

#### `get_mri_sublist.py`：从 fmriprep 输出目录列出被试
```bash
python src/preprocess/get_mri_sublist.py --dir <fmriprep_rest_dir> --out <mri_sublist.txt>
```
关键参数：
- `--dir`: fmriprep 目录（包含 `sub-*`）
- `--out`: 输出 txt 路径

#### `screen_head_motion_efny.py`：统计 rest FD，并标注有效 run
```bash
python src/preprocess/screen_head_motion_efny.py --fmriprep-dir <fmriprep_rest_dir> --out <rest_fd_summary.csv>
```
关键参数：
- `--fmriprep-dir`: fmriprep rest 目录
- `--out`: 输出 CSV（含 `valid_subject` 与 `meanFD`）

#### `generate_valid_sublists.py`：从 QC 表生成有效被试列表
无命令行参数，直接运行：
```bash
python src/preprocess/generate_valid_sublists.py
```
注意：数据根目录在脚本内常量 `DATA_ROOT`。

#### `build_behavioral_data.py`：合并 demo 与 metrics，生成 `EFNY_behavioral_data.csv`
```bash
python src/preprocess/build_behavioral_data.py --metrics <EFNY_metrics.csv> --demo <EFNY_demo_with_rsfmri.csv> --output <EFNY_behavioral_data.csv>
```
关键参数：
- `--metrics/-m`: 行为 metrics 表
- `--demo/-d`: demo 表
- `--output/-o`: 输出路径
- `--log/-l`: log 路径

### 2) 功能连接（FC）计算与向量化（`src/functional_conn`）

#### `compute_fc_schaefer.py`：计算单被试 FC 矩阵（CSV）
```bash
python src/functional_conn/compute_fc_schaefer.py --subject <sub-xxx> --n-rois 100
```
关键参数：
- `--xcpd-dir`: xcp-d 输出根目录
- `--subject`: 被试 ID（如 `sub-xxx`）
- `--n-rois`: Schaefer 分区数（100/200/400）
- `--qc-file`: QC CSV（决定 valid runs）
- `--valid-list`: 有效被试列表
- `--out-dir`: 输出目录

#### `fisher_z_fc.py`：FC 做 Fisher-Z 变换（CSV）
```bash
python src/functional_conn/fisher_z_fc.py --subject <sub-xxx> --n-rois 100
```
关键参数：
- `--in-dir`: FC 输入目录
- `--out-dir`: Z 后输出目录
- `--subject`, `--n-rois`

#### `convert_fc_vector.py`：将 FC_z 矩阵下三角向量化（npy）
该脚本参数较多（输入/输出/被试列表/atlas 分辨率等），建议直接查看：
```bash
python src/functional_conn/convert_fc_vector.py --help
```

#### `compute_group_avg_fc.py`：组平均 FC（可选可视化）
```bash
python src/functional_conn/compute_group_avg_fc.py --in-dir <fc_dir> --sublist <rest_valid_sublist.txt> --visualize
```
关键参数：
- `--in-dir`: 包含 `Schaefer*` 子目录
- `--sublist`: 被试列表
- `--atlas` / `--n-rois`: 指定 atlas
- `--out-dir`: 输出目录

#### `plot_fc_matrix.py`：矩阵可视化（支持 Yeo17 排序）
```bash
python src/functional_conn/plot_fc_matrix.py --file <matrix.csv> --out <out.png> --title "..." --yeo17 --n-rois 100
```
关键参数：
- `--file`: 输入 CSV
- `--out`: 输出 PNG
- `--yeo17`: 以 Yeo17 网络排序并加边界线（需要 `--n-rois`）

### 3) 行为指标（`src/metric_compute`）

#### `compute_efny_metrics.py`：从 app data 计算行为指标
该脚本没有命令行参数，路径在脚本顶部常量中定义（`DATA_DIR` / `TASK_CSV` / `OUT_CSV`）。
```bash
python src/metric_compute/compute_efny_metrics.py
```

#### `metrics_similarity_heatmap.py`：行为指标相关热图
```bash
python src/metric_compute/metrics_similarity_heatmap.py --csv <EFNY_metrics.csv> --task-csv <EFNY_task.csv> --out-png <out.png>
```
关键参数：
- `--method`: pearson/spearman/kendall
- `--min-valid-ratio`: 单列有效数据占比阈值
- `--min-pair-ratio`: 两列共同有效数据占比阈值

## 脑-行为关联分析（`src/scripts/run_single_task.py`）

该入口脚本支持：真实分析（`task_id=0`）与单次置换（`task_id>=1`）。

### 1) 最常用命令
```bash
# 真实数据
python src/scripts/run_single_task.py --task_id 0 --model_type adaptive_pls --config_file src/models/config.json

# 单次置换（种子由 task_id 决定，便于 HPC array）
python src/scripts/run_single_task.py --task_id 1 --model_type adaptive_pls --config_file src/models/config.json
```

### 2) 关键参数速查
- `--task_id`: 0=真实；1..N=置换
- `--model_type`: `adaptive_pls` / `adaptive_scca` / `adaptive_rcca`
- `--config_file`: 配置文件（建议使用 `src/models/config.json`）
- `--random_state`: 随机种子（真实重复跑时常用）
- `--covariates_path`: 可选，协变量 CSV（需包含 `age/sex/meanFD`）
- `--cv_n_splits`: 外层 CV 折数（默认 5）
- `--max_missing_rate`: 缺失率阈值
- `--output_dir`: 输出根目录（默认写入项目 results 目录）
- `--output_prefix`: 输出前缀
- `--save_formats`: `json` / `npz`
- `--log_level`, `--log_file`

查看完整参数：
```bash
python src/scripts/run_single_task.py --help
```

### 3) 嵌套交叉验证（推荐框架）
目标：外层评估泛化能力，内层选择超参数，所有预处理严格在训练折内完成以避免信息泄露。

推荐流程（每次运行都遵循同一逻辑）：
1. 外层 KFold（n_outer）划分训练/测试。
2. 对每个外层折：
   - 只用外层训练集拟合预处理：缺失值填补均值、协变量回归、标准化、可选 PCA。
   - 在外层训练集上做内层 KFold（n_inner）：
     * 对每组候选参数：在内层训练折拟合同样的预处理，并在内层验证折评估。
     * 使用统一指标选参（推荐：`meancorr`，即各成分相关系数的均值）。
   - 选择内层平均分最高的参数；如有并列，可用更小方差或更少参数作为 tie-break。
   - 用最佳参数在“外层训练集”重新拟合模型（固定参数，不再做模型内部 CV），再在“外层测试集”评估。
3. 汇总外层结果：各成分相关系数的均值/方差等整体指标。
4. 置换检验：每次置换仅打乱 Y（保持 X/协变量索引一致），完整重复上述嵌套流程并记录种子。

实现约定（便于后续修改）：
- 预处理步骤的 fit 只发生在训练折（外层与内层都遵守）。
- 模型内部不再做选参型 CV；超参数仅由内层 CV 决定。
- 输出至少包含：`outer_fold_results`、`inner_cv_table`、`outer_mean_canonical_correlations`、`outer_all_test_canonical_correlations`、随机种子与参数网格规模。

### 4) HPC（SLURM）
仓库提供了 3 个示例提交脚本（可按需要改 `MODEL_TYPE`、array 范围、log 路径等）：
```bash
sbatch src/scripts/submit_hpc_real.sh   # 多次真实运行（array=0-10）
sbatch src/scripts/submit_hpc_perm.sh   # 置换运行（array=1-1000）
sbatch src/scripts/submit_hpc_job.sh    # 0=真实，1..N=置换（单脚本）
```

## 结果汇总（real/perm 扫描）

### `src/result_summary/summarize_real_perm_scores.py`
扫描 `results_root/real` 与 `results_root/perm`，提取每次运行的相关向量并输出 CSV。

```bash
python src/result_summary/summarize_real_perm_scores.py --results_root <results_root> --analysis_type both --atlas <atlas> --model_type <model>
```

关键参数：
- `--results_root`: 结果根目录
- `--analysis_type`: real / perm / both
- `--atlas`: 可选过滤
- `--model_type`: 可选过滤
- `--output_csv`: 输出 CSV
- `--score_mode`: `first_component` / `mean_all`

## 输出位置（约定）

常用输出：
- QC / 表格：`data/EFNY/table/...`
- FC 矩阵与向量：`data/EFNY/functional_conn...`
- 脑-行为关联：默认写入 `results/real/...` 与 `results/perm/...`（可用 `--output_dir` 改写）
- 大型数组（如每折 `X_scores/Y_scores`）会保存到同目录的 `artifacts/`，JSON/NPZ 内仅保留索引与路径
