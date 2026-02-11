# visit3 序列导出与 app_sequence/visit3 一致性比对（基于 item_group_002 + answer_group_002）

## 目的

根据确认规则：

- `match` 采用 `Left`；
- `item_group_002` 应匹配 `answer_group_002`；
- `answer_group_002` 与 `answer_group_004` 在 `Spatial1Back`、`Spatial2Back`、`Emotion2Back` 上存在答案差异。

据此将 `item_group_002`（item）与 `answer_group_002`（answer）导出为 visit3 候选序列，并与 `data/raw/behavior_data/app_sequence/visit3` 对比。

## 生成方式

执行脚本：

- `python -m scripts.build_visit3_library_v2 --dataset EFNY_THU --config configs/paths.yaml`

输出目录：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/`

主要输出文件：

- `visit3_item_group2_templates.json`
- `visit3_answer_group2_templates.json`
- `visit3_from_groups.json`
- `visit3_from_app_sequence.json`
- `visit3_compare_to_app_sequence.json`

## 比对结果摘要

### item 序列（groups vs app_sequence/visit3）

- 完全一致任务数：0
- 可比但不一致任务数：10  
  - `CPT`、`ColorStroop`、`DCCS`、`DT`、`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`FZSS`、`Spatial1Back`、`Spatial2Back`
- app_sequence 缺失或不可比任务数：8  
  - `EmotionSwitch`、`FLANKER`、`GNG`、`KT`、`Number1Back`、`Number2Back`、`SST`、`ZYST`

### answer 序列（groups vs app_sequence/visit3）

- 完全一致任务数：0
- 可比但不一致任务数：13（全部可比任务均未达到完全一致）
- app_sequence 缺失或不可比任务数：5  
  - `EmotionSwitch`、`FLANKER`、`GNG`、`Number1Back`、`Number2Back`

## 结论

基于 `item_group_002 + answer_group_002` 导出的 visit3 序列，**不匹配** `data/raw/behavior_data/app_sequence/visit3`。

更具体地说：

- item 侧在可比任务中无一完全一致；
- answer 侧在可比任务中同样无一完全一致；
- 且 `app_sequence/visit3` 本身任务覆盖不完整（仅 13 任务），无法覆盖当前分组导出的完整 18 任务模板。

因此，当前流程下更应将 `app_sequence/visit3` 视为参考信息，而非 visit3 真值序列来源。

