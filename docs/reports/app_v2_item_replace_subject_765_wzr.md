# APP v2：THU_20250921_765_WZR item 序列更新为 visit3

## 背景

- 被试：`THU_20250921_765_WZR`
- 依据：该被试属于 answer `group_002`，因此 item 应与 `visit3` 一致。

## 执行

命令：

```bash
python -m scripts.app_v2_apply_visit_items_to_subject --dataset EFNY_THU --subject-id THU_20250921_765_WZR --visit-template visit3_from_groups.json
```

脚本：

- `scripts/app_v2_apply_visit_items_to_subject.py`

## 数据来源

- item 模板：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit3_from_groups.json`
- 原始工作簿：`data/raw/behavior_data/cibr_app_data/THU_20250921_765_WZR_王泽锐_GameData.xlsx`

## 结果

- 已完成 item 列覆盖（`正式阶段刺激图片/Item名`）。
- 该工作簿包含 3 个任务 sheet，本次实际更新 3 个任务（`updated_tasks=3`）。
- 未出现缺失模板或缺失 item 列。

## 备份

- 替换前备份目录：`data/raw/behavior_data/cibr_app_data/_backup_item_replace_THU_20250921_765_WZR_20260212_172604`

## 备注

- 本次仅更新 item，不改写 answer。
- 本次未重跑 item 分组；若需将该被试在 `groups_by_item` 中重新归组，可在下一步执行分组重建或定向迁移。
