# 工作流（工程）

本文件描述在 `PROJECT_STRUCTURE.md` 固定目录结构下，脚本入口的标准化调用方式。

## 标准 CLI 形式

`scripts/` 下所有可运行入口应接受：

- `--dataset <DATASET_NAME>`
- `--config <PATH_TO_CONFIGS_PATHS_YAML>`

示例：

```bash
python -m scripts.run_single_task --dataset EFNY --config configs/paths.yaml --dry-run
```

数据集相关假设配置在 `configs/datasets/<DATASET>.yaml`。
数据集无关的分析默认值可配置在 `configs/analysis.yaml`。

## data/ 与 outputs/ 约定

- `data/raw/<DATASET>/`: 原始输入（不由流水线脚本生成）
- `data/interim/<DATASET>/`: 中间产物（例如 MRI 预处理或连接组中间结果）
- `data/processed/<DATASET>/`: 可复用的处理后数据（如表格、FC 向量特征）
- `outputs/<DATASET>/`: 运行产物（结果、图表）
- `outputs/<DATASET>/logs/`: 运行日志（脚本日志、SLURM stdout/stderr）

部分外部输入（如 fMRIPrep 输出）可能位于仓库外；请在 `configs/datasets/<DATASET>.yaml` 的 `external_inputs` 下配置绝对路径。

## 结果目录

`src/result_summary/` 下的汇总脚本默认使用：

- `outputs/<DATASET>/results`

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

这些脚本将 `#SBATCH --chdir` 设为集群项目根目录，并将 SLURM stdout/stderr 与任务日志写入 `outputs/<DATASET>/logs/...`。
注意：SLURM 的 `#SBATCH --output/--error` 路径为静态字符串，无法展开环境变量，因此在脚本头部保持数据集特定路径。

示例：

```bash
sbatch scripts/submit_hpc_real.sh
```
