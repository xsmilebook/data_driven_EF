# APP grouping outputs audit

本报告用于检查 `data/interim/behavioral_preprocess/groups_by_answer/` 与 `.../groups_by_item/`
中由 `temp/group_app_stimulus_groups.py` 生成的分组结果是否与当前原始数据目录一致。

- dataset: `EFNY_THU`
- raw app dir: `data/raw/behavior_data/cibr_app_data`

## Summary

| kind | n_groups | total_lines | unique_subjects | duplicate_lines | missing_in_raw |
| --- | --- | --- | --- | --- | --- |
| answer | 9 | 646 | 645 | 1 | 7 |
| item | 4 | 646 | 645 | 1 | 7 |

## Details

- answer group sizes: group_001_sublist:316, group_002_sublist:140, group_003_sublist:81, group_004_sublist:36, group_005_sublist:32, group_006_sublist:25, group_007_sublist:8, group_008_sublist:5, group_009_sublist:3
- item group sizes: group_001_sublist:398, group_002_sublist:230, group_003_sublist:16, group_004_sublist:2

- duplicated subject_id lines (should be 0):
  - THU_20250725_684_LC

- subject_ids present in grouping outputs but missing from current raw dir:
  - THU_20231217_144_LSC (date=2023-12-17)
  - THU_20231230_161_ZJY (date=2023-12-30)
  - THU_20240120_183_DKZ (date=2024-01-20)
  - THU_20241130_435_CJT (date=2024-11-30)
  - THU_20250605_601_ZYY (date=2025-06-05)
  - THU_20250613_602_ZWZ (date=2025-06-13)
  - THU_20250709_628_ZSY (date=2025-07-09)

备注：若 `missing_in_raw>0`，通常意味着分组结果是在去重/移除重复工作簿之前生成的，
需要在当前 `data/raw/behavior_data/cibr_app_data/` 基础上重新生成分组结果。

