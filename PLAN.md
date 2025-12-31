# PLAN.md

`data_driven_EF` 的单数据集（EFNY）重构计划与执行记录。

## Project objectives and success criteria

- 取消 `data/` 与 `outputs/` 的数据集层级，仅保留 EFNY 数据流程。
- 路径解析与脚本默认值可在单数据集结构下运行。
- 文档与结构说明同步更新。

## Technical requirements and constraints

- 不修改 `data/` 与 `outputs/` 运行产物内容。
- 不改动科学/分析逻辑，仅调整路径与配置。
- 保持脚本入口可用（必要时保留 `--dataset` 参数以兼容）。

## Architecture decisions and rationale

- 统一路径根：`data/raw`, `data/interim`, `data/processed`, `outputs`。
- 结果与日志集中在 `outputs/results` 与 `outputs/logs`。
- EFNY 特定假设集中在 `configs/paths.yaml` 的 `dataset` 段落。

### Atlas selection (Schaefer)

继续通过 `configs/paths.yaml` 的 `dataset.files.brain_file` 路径切换 Schaefer100/200/400。

## Feature roadmap and development phases

### Phase A: Path and entry-point standardization (completed)

### Phase B: Documentation and workflow alignment (completed)

### Phase C: Import hygiene and thin entry points (completed)

## Current action items (from create-plan)

- [ ] 确认所有脚本在单数据集路径下可解析输入/输出（必要时补充默认值）。
- [ ] 如需兼容旧路径，补充最小化迁移说明。

## Testing and deployment strategy

- 运行 `python -m scripts.run_single_task --dry-run` 验证路径解析与导入。
