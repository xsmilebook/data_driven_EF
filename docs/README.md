# 文档索引

本目录包含 `data_driven_EF` 项目的人类可读文档。

## 关键文件

- `docs/workflow.md`: 如何运行流程。
- `docs/methods.md`: 方法学说明（面向论文；可能不完整）。
- `docs/data_dictionary.md`: 术语/字段说明（仍在完善）。
- `docs/reports/`: 研究计划与总结。
- `docs/reports/visit_sequence_changes.md`: APP visit 序列改动说明（visit1–visit4）。
- `docs/reports/visit1_consistency_20231014_20250707.md`: 20231014–20250707 期间与 visit1 一致性检查。
- `docs/reports/app_grouping_audit.md`: APP 序列分组产物（data/interim）一致性审计。
- `docs/reports/app_data_cleaning_v2_workflow.md`: APP 数据清洗 v2 流程草案（按日期顺序 + 序列类型）。
- `docs/reports/visit1_items_template_vs_app_sequence.md`: 对比 visit1 的 group 模板与 app_sequence 配置的 items 一致性（用于检查配置解析与命名映射）。
- `docs/reports/visit1_answer_groups_within_item_group1.md`: 在 item group_001 内检查 answer group 分布，并定位答案差异集中任务（用于确认 visit1 答案基线）。
- `docs/reports/item_group_sequence_changes.md`: item 分组（group_001–group_004）刺激序列差异分析。
- `docs/reports/app_v2_item_replace_answer_group1.md`: 将 answer_group1 中 item≠item_group1 的被试替换为 item_group1 模板（网站崩溃期参考被试填充错误修正）。
- `docs/reports/visit3_group2_export_comparison.md`: 基于 item_group_002 + answer_group_002 导出 visit3 序列，并与 `app_sequence/visit3` 做一致性比对。
- `docs/reports/visit3_v1_sequence_parsing.md`: 解析 `visit3_v1`（非规范 Excel）为 JSON，并与 `answer_group_004` 进行对照判断。
- `docs/reports/visit3_v1_from_groups_build.md`: 在 `run_corrected_v2` 中按规则生成 `visit3_v1`（Spatial1Back/Spatial2Back 来自 `visit3_v1`，Emotion2Back 使用 `answer_group_004` 答案且 item 置空）。
- `docs/reports/app_v2_item_replace_answer_group004.md`: 将 `answer_group_004` 全部被试替换为 `visit3_v1` item 序列，并将其从 item `group_002` 迁移至 item `group_005`（含原始 xlsx 备份路径与分组核验）。
- `docs/reports/app_v2_item_replace_subject_765_wzr.md`: 将 `THU_20250921_765_WZR` 的 item 序列更新为 `visit3`（因其 answer group 为 002），并记录备份路径与执行结果。
- `docs/reports/app_v2_conformed_sequences.md`: 统一导出 `run_corrected_v2` 中已确认正确的序列 JSON 到 `conformed/`，固定命名与来源映射。
- `docs/reports/app_v2_visit2_confirmed_build.md`: 基于 `visit2-1125` + `visit3_confirmed` 生成 `visit2_confirmed.json` 的规则、字段映射与核验结果。

## Notes 与 sessions 的区别

- `docs/sessions/`: 按时间记录的会话日志（AI/开发会话）。
- `docs/notes/`: 用户随手记录与自由想法。

## 日志

- 所有运行时日志存放在 `outputs/logs/`。
- 按任务组织日志文件（如 `outputs/logs/real_adaptive_pls/`）。
