# APP v2：visit3_v2_confirmed 生成与 answer_group008 item 更新

## 目标

基于已确认任务来源规则生成 `visit3_v2_confirmed`，并用于更新 `answer_group008` 被试在原始 xlsx 中的 item 列；同时完成 item 分组迁移。

## 规则（任务 -> 来源）

- `visit3_confirmed`：`CPT, ColorStroop, DT, EmotionStroop, EmotionSwitch, FLANKER, FZSS, GNG, KT, Emotion1Back, Number1Back, Number2Back`
- `visit3_v1_confirmed`：`Emotion2Back, Spatial1Back`
- `visit2_confirmed`：`SST`
- `visit1_confirmed`：`DCCS`
- `app_sequence/visit3_v2/spatialNback2.1.json`：`Spatial2Back`

## 执行脚本

1) 生成 `visit3_v2_confirmed`：

```bash
python -m scripts.build_visit3_v2_confirmed_v2 --dataset EFNY_THU
```

2) 用该版本替换 `answer_group008` 的 raw item，并迁移 item 分组：

```bash
python -m scripts.app_v2_apply_visit3_v2_items_to_answer_group008 --dataset EFNY_THU
```

## 输出与变更

- 新增/更新（conformed）：
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit3_v2_confirmed.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit3_v2_confirmed_build_meta.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/manifest.json`

- 原始 xlsx item 批量更新：
- 目标被试：`answer_group008` 共 5 人
- 备份目录：`data/raw/behavior_data/cibr_app_data/_backup_item_replace_answer_group008_visit3v2_20260213_140306`
- 更新结果：`updated_subjects_n=5`, `missing_workbooks_n=0`

- item 分组迁移：
- 从 `group_003` 移除上述 5 人（16 -> 11）
- 加入 `group_004`（0 -> 5）
- 重写 `groups_by_item/groups_manifest.json` 与 `groups_manifest.csv`

## 快速核验

- 抽查 `THU_20251207_806_ZYW`：
- `DCCS` item 差异 `0/40`
- `SST` item 差异 `0/97`
- `Spatial2Back` item 差异 `0/60`
