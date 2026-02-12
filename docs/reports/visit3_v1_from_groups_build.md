# run_corrected_v2 生成 visit3_v1 序列（基于确认规则）

## 目标

在 `run_corrected_v2` 中生成 `visit3_v1` 序列版本，并执行以下规则：

- 除 `Spatial1Back`、`Spatial2Back`、`Emotion2Back` 外，其余任务与现有 `visit3` 序列保持一致（基于 `visit3_from_groups.json`）。
- `Spatial1Back`、`Spatial2Back`：使用 `data/raw/behavior_data/app_sequence/visit3_v1/` 中序列填充 item 与 answer。
- `Emotion2Back`：item 全部置空；answer 使用 `answer_group_004` 的序列。

## 实现与输入

脚本：

- `python -m scripts.build_visit3_v1_library_v2 --dataset EFNY_THU --config configs/paths.yaml`

输入：

- 基线 visit3：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_from_groups.json`
- visit3_v1 空间任务：  
  - `data/raw/behavior_data/app_sequence/visit3_v1/spatial1back2.1.json`
  - `data/raw/behavior_data/app_sequence/visit3_v1/spatial2back2.1.json`
- `Emotion2Back` 答案来源：`answer_group_004` 被试中 `Emotion2Back` 非空答案最长者

## 输出

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_v1_from_groups.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_v1_build_meta.json`

其中 `visit3_v1_build_meta.json` 记录了规则与 `Emotion2Back` 答案来源被试信息。

## 结果核验

与 `visit3_from_groups.json` 对比：

- 仅以下 3 个任务发生变化：`Spatial1Back`、`Spatial2Back`、`Emotion2Back`
- 其余任务保持不变（15 个任务）

关键字段检查：

- `Spatial1Back`: `items_len=60`, `answers_len=60`
- `Spatial2Back`: `items_len=60`, `answers_len=60`
- `Emotion2Back`: `items_len=58`, `answers_len=58`, 且 `items_none_n=58`（全部为空）

`Emotion2Back` 答案来源被试：

- `THU_20250708_623_WYR`

