# v2 最小开发骨架（基于已清洗数据）

本文档定义“先冻结 v1、再基于清洗结果重开 v2”的最小可运行骨架。

## 1) 冻结与存档

新增命令入口：

```bash
python -m scripts.archive_legacy_snapshot --archive-name v1_freeze_20260219
```

默认仅预演，不执行写操作。  
实际执行（创建分支 + tag）：

```bash
python -m scripts.archive_legacy_snapshot --archive-name v1_freeze_20260219 --execute
```

需要同时推送到远端：

```bash
python -m scripts.archive_legacy_snapshot --archive-name v1_freeze_20260219 --push --execute
```

脚本行为：
- 读取当前分支与 HEAD。
- 默认创建：`archive/<archive_name>` 分支与 `archive/<archive_name>` 注释 tag。
- 工作区不干净时默认拒绝执行（可加 `--allow-dirty` 放行）。

## 2) v2 最小入口

新增命令入口：

```bash
python -m scripts.v2_run_pipeline --dataset EFNY --config configs/paths.yaml --dry-run
```

默认从 `configs/paths.yaml -> dataset.files.behavioral_metrics_file` 读取清洗后行为表；  
可手动覆盖：

```bash
python -m scripts.v2_run_pipeline \
  --dataset EFNY \
  --config configs/paths.yaml \
  --behavior-table data/processed/table/metrics/EFNY_beh_metrics.csv \
  --run-id pilot_v2_001
```

当前最小能力：
- 统一解析配置与路径（dataset/config）。
- 读取并检查清洗后行为表。
- 产出 profile 元信息（样本数、列数、缺失率、被试列检测）。

## 3) 代码骨架位置

- 入口：`scripts/v2_run_pipeline.py`
- 核心：`src/models/v2_pipeline.py`

当前不做完整建模迁移，仅保证 v2 流水线从“清洗完成数据”稳定起步。

