# APP 分组重建与匹配审计（2026-02-13）

本次基于当前原始目录 `data/raw/behavior_data/cibr_app_data/` 重新生成 item/answer 分组，
并在不覆盖原有产物的前提下，完成 item 组与 answer 组的一致性审计。

## 1. 产物位置（新增，不覆盖旧分组）

- answer 新分组：`data/interim/behavioral_preprocess/groups_by_answer_regen_20260213/`
- item 新分组：`data/interim/behavioral_preprocess/groups_by_item_regen_20260213/`
- 匹配审计：`data/interim/behavioral_preprocess/grouping_audit_regen_20260213/`

其中审计目录包含：

- `subject_item_answer_mapping.csv`：被试级 item/answer 组映射；
- `item_answer_crosstab.csv`：item 组 × answer 组列联表；
- `summary.json`：汇总统计与映射关系。

## 2. 重建结果

- 输入工作簿总数：639
- answer 分组数：7
- item 分组数：6
- 两侧共同覆盖被试：639（item-only=0，answer-only=0）

## 3. 匹配审计结论

### 3.1 主体结论

- item 与 answer 分组在“被试覆盖”层面完全一致（无缺失、无孤立被试）。
- 分组标签编号不完全同名，但形成稳定映射关系。

### 3.2 item->answer 映射（按新分组）

- `item group_001` -> `answer group_001`（342） + `answer group_003`（105）
- `item group_002` -> `answer group_002`（139）
- `item group_003` -> `answer group_004`（36）
- `item group_004` -> `answer group_005`（8）
- `item group_005` -> `answer group_006`（5）
- `item group_006` -> `answer group_007`（4）

上述关系与近期人工修正后的版本演化一致：`item group_001` 内部存在两个 answer 子型，其余 item 组与 answer 组为一对一对应。

## 4. 复现命令

```powershell
python temp/group_app_stimulus_groups.py --grouping answer --out-dir data/interim/behavioral_preprocess/groups_by_answer_regen_20260213
python temp/group_app_stimulus_groups.py --grouping item --out-dir data/interim/behavioral_preprocess/groups_by_item_regen_20260213
```

