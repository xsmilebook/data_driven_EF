# visit3_v1 序列解析与 answer_group_004 对照（Spatial1Back/Spatial2Back/Emotion2Back）

## 目的

`visit3_v1` 中 `Spatial2Back` 与 `Emotion2Back` 的文件格式不规范，先解析为标准 JSON，再与 `answer_group_004` 的被试答案进行对照判断。

## 原始文件结构（visit3_v1）

来源目录：`data/raw/behavior_data/app_sequence/visit3_v1/`

- `emotion2back2.1.xlsx`
- 单 sheet（`Sheet1`）
- 列：`picName`、`buttonName`、`配对错误`
- `buttonName` 中存在字符串 `None`，需归一化为空值

- `SpaticalNback_旧序列.xlsx`
- `Sheet1` 有效，`Sheet2/Sheet3` 为空
- `Sheet1` 中：
  - B 列为 item（如 `010000000`、`000000100`）
  - C 列为答案（`Left/Right` 或字符串 `None`）
  - 需要去前导 0 归一化

## 解析输出（JSON）

已解析为 JSON 并写入 v2 序列库目录：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_v1_parsed.json`

结构示例（任务 → items/answers）：

- `Emotion2Back.items/answers`（来自 `emotion2back2.1.xlsx`）
- `Spatial1Back.items/answers`（来自 `SpaticalNback_旧序列.xlsx`）
- `Spatial2Back.items/answers`（来自 `SpaticalNback_旧序列.xlsx`）

## 与 answer_group_004 对照

选择 `answer_group_004` 中一名被试：`THU_20250707_622_JHY`  
对比输出写入：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_v1_compare_to_group004.json`

结果（以 `正式阶段正确答案` 为准）：

- `Spatial2Back`：完全一致（match_ratio = 1.0，len=58）
- `Spatial1Back`：部分一致（match_ratio = 0.6897，len=59 vs 58）
- `Emotion2Back`：无法判断（被试答案为空，visit3_v1 有 58 个答案）

## 结论

- `visit3_v1` 的 `Spatial2Back` 与 `answer_group_004` 的答案一致性高（示例被试为完全一致）。
- `Spatial1Back` 仅部分一致，提示 `visit3_v1` 与 `answer_group_004` 之间存在序列差异或对齐问题。
- `Emotion2Back` 在 `answer_group_004` 中答案缺失，无法判断其与 `visit3_v1` 的匹配程度。

