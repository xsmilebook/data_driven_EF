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

## Notes 与 sessions 的区别

- `docs/sessions/`: 按时间记录的会话日志（AI/开发会话）。
- `docs/notes/`: 用户随手记录与自由想法。

## 日志

- 所有运行时日志存放在 `outputs/logs/`。
- 按任务组织日志文件（如 `outputs/logs/real_adaptive_pls/`）。
