# APP v2：visit1_v1_confirmed 生成与 answer_group009 item 更新

## 目标

基于已确认任务来源规则生成 `visit1_v1_confirmed`，并用于更新 `answer_group009` 被试在原始 xlsx 中的 item 列；同时完成 item 分组迁移。

## 规则（任务 -> 来源）

- `visit1_confirmed`：`CPT, DCCS, DT, EmotionSwitch, FLANKER, FZSS, GNG, KT, Number1Back, Number2Back, SST, Spatial1Back, Spatial2Back, ZYST`
- `visit3_confirmed`：`ColorStroop, Emotion1Back, EmotionStroop`
- `visit3_v1_confirmed`：`Emotion2Back`

## 执行脚本

1) 生成 `visit1_v1_confirmed`：

```bash
python -m scripts.build_visit1_v1_confirmed_v2 --dataset EFNY_THU
```

2) 用该版本替换 `answer_group009` 的 raw item，并迁移 item 分组：

```bash
python -m scripts.app_v2_apply_visit1_v1_items_to_answer_group009 --dataset EFNY_THU
```

## 输出与变更

- 新增/更新（conformed）：
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit1_v1_confirmed.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit1_v1_confirmed_build_meta.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/manifest.json`

- 原始 xlsx item 批量更新：
- 目标被试：`answer_group009` 共 3 人
- 备份目录：`data/raw/behavior_data/cibr_app_data/_backup_item_replace_answer_group009_visit1v1_20260213_144808`
- 更新结果：`updated_subjects_n=3`, `missing_workbooks_n=0`

- item 分组迁移：
- 从 `group_003` 移除上述 3 人（11 -> 8）
- 加入 `group_006`（0 -> 3）
- 重写 `groups_by_item/groups_manifest.json` 与 `groups_manifest.csv`

## 快速核验

- 抽查 `THU_20251220_814_LCY`：
- `ColorStroop` item 差异 `0/96`
- `Emotion2Back` item 差异 `0/60`
- `SST` item 差异 `0/97`
- `Spatial2Back` item 差异 `0/60`
