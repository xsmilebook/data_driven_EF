# APP v2：将 answer_group2 的错误 item 序列替换为 item_group1 模板

## 背景与假设

在网站崩溃期，部分被试的 APP 数据是通过设备端 txt 转换为 xlsx 的方式补导出。该流程需要选择“参考被试”来填充/对齐刺激列（`正式阶段刺激图片/Item名`），若参考被试选择错误，会导致这些被试的 **item 序列错误**，而答案等设备端字段仍可能是正确的。

基于上述假设，本次对 **answer_group2** 中所有“item 序列与 item_group1 不一致”的被试，将其工作簿内的 item 列统一替换为 **item_group1（group_001）模板**。

## 输入与依据

- 目标被试集合：`data/interim/behavioral_preprocess/groups_by_answer/group_002_sublist.txt`
- item 模板来源：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit1_item_group1_templates.json`
- 修改对象（原始工作簿）：`data/raw/behavior_data/cibr_app_data/*_GameData.xlsx`

## 操作内容

对每个目标被试的工作簿：

- 在每个任务 sheet 中定位 `正式阶段刺激图片/Item名` 列；
- 用 `visit1_item_group1_templates.json` 中对应任务的 item 模板逐行覆盖该列；
- 不修改答案列（`正式阶段正确答案`）及其他字段；
- 为避免不可逆风险，先对原始工作簿做备份拷贝（见下）。

执行脚本：

- `python -m scripts.app_v2_replace_items_for_answer_group2 --dataset EFNY_THU --config configs/paths.yaml`

备份目录（本次运行生成）：

- `data/raw/behavior_data/cibr_app_data/_backup_item_replace_20260128_152416/`

同时，为使中间产物与最新 raw 数据一致，已重新生成 item 分组（并归档旧版本）：

- 旧分组归档：`data/interim/behavioral_preprocess/_archived_groupings/20260128_152547/groups_by_item/`
- 新分组输出：`data/interim/behavioral_preprocess/groups_by_item/`

## 结果摘要

- answer_group2 被试数：139
- 被替换的被试数：139
- 替换后，这 139 名被试在 item 分组上全部归入 item_group1（group_001）：
  - 替换后 `item_group1` 总人数：530（较替换前增加 139）

## 备注与限制

- 本次仅修正 item 列（刺激序列）；不对答案序列做任何“真值修正”。
- `SST` 的 item 在工作簿中通常缺失（近似空序列），因此替换对 SST 的直接可观测影响有限；但仍保持统一覆盖逻辑以便审计。

