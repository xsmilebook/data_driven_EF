# APP v2：answer_group_004 的 item 替换与 item 分组迁移

## 目标

按已确认规则执行两步修正：

- 将 `answer_group_004` 全部被试的原始 APP xlsx 的 item 序列替换为 `visit3_v1`；
- 将这些被试从 item `group_002` 迁移到 item `group_005`。

同时保留替换前原始文件备份，确保可追溯。

## 执行脚本

```bash
python -m scripts.app_v2_apply_visit3_v1_items_to_answer_group004 --dataset EFNY_THU
```

脚本：

- `scripts/app_v2_apply_visit3_v1_items_to_answer_group004.py`

## 输入与规则

- 被试来源：`data/interim/behavioral_preprocess/groups_by_answer/group_004_sublist.txt`
- item 模板：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_v1_from_groups.json`
- 写入目标：`data/raw/behavior_data/cibr_app_data/*.xlsx` 中对应被试文件的 `正式阶段刺激图片/Item名` 列
- 分组迁移：
- 从 `data/interim/behavioral_preprocess/groups_by_item/group_002_sublist.txt` 删除上述被试
- 写入 `data/interim/behavioral_preprocess/groups_by_item/group_005_sublist.txt`
- 重写 `groups_by_item/groups_manifest.json` 与 `groups_manifest.csv`

## 备份与日志

- 原始 xlsx 备份目录：`data/raw/behavior_data/cibr_app_data/_backup_item_replace_answer_group004_visit3v1_20260212_170032`
- 本次运行日志（临时）：`temp/app_v2_apply_visit3_v1_group004_log.json`

## 结果核验

- `answer_group_004` 被试数：36
- 成功更新 xlsx：36
- 缺失工作簿：0
- 迁移后 item 分组计数：
- `group_002`: 138（由 174 减少 36）
- `group_005`: 36（新建）
- 集合一致性：
- `group_005` 与 `answer_group_004` 完全一致（双向差集均为 0）
- `answer_group_004` 与 `group_002` 交集为 0

## 备注

- 该操作只替换 item 列，不改写 `正式阶段正确答案`。
- `group_005` 在 manifest 中标记为 `manual_group_assignment`，用于区分人工确认迁移结果。
