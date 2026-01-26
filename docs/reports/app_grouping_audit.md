# APP grouping outputs audit

本报告用于检查 `data/interim/behavioral_preprocess/groups_by_answer/` 与 `.../groups_by_item/`
中由 `temp/group_app_stimulus_groups.py` 生成的分组结果是否与当前原始数据目录一致。

- dataset: `EFNY_THU`
- raw app dir: `data/raw/behavior_data/cibr_app_data`

## Summary

| kind | n_groups | total_lines | unique_subjects | duplicate_lines | missing_in_raw |
| --- | --- | --- | --- | --- | --- |
| answer | 9 | 638 | 638 | 0 | 0 |
| item | 4 | 638 | 638 | 0 | 0 |

## Details

- answer group sizes: group_001_sublist:311, group_002_sublist:139, group_003_sublist:80, group_004_sublist:36, group_005_sublist:31, group_006_sublist:25, group_007_sublist:8, group_008_sublist:5, group_009_sublist:3
- item group sizes: group_001_sublist:391, group_002_sublist:229, group_003_sublist:16, group_004_sublist:2

备注：若 `missing_in_raw>0`，通常意味着分组结果是在去重/移除重复工作簿之前生成的，
需要在当前 `data/raw/behavior_data/cibr_app_data/` 基础上重新生成分组结果。

