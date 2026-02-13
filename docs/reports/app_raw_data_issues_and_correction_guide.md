# APP 原始数据问题与修正流程说明（`cibr_app_data`）

本文整理 `data/raw/behavior_data/cibr_app_data` 在当前阶段已识别的问题类型、对应修正规则、执行步骤与产物位置，用于后续复现与审计。

## 1. 数据问题概览

### 1.1 文件层问题

- 重复工作簿：同一被试存在多个 `*_GameData.xlsx`。
- 导出来源异质：Web 导出与设备端 txt 转换后 xlsx 混用，字段完整性不一致。
- 个别被试任务数量异常：存在 17/19 sheet 的非标准工作簿。

### 1.2 试次层问题

- `SST` 异常行：存在 97 行（应按规则仅保留前 96 trial）。
- `EmotionStroop` 缺失答案：部分被试在关键 trial 出现空值（本次已按规则补全）。
- NBack 前导空值表示不统一：`None`、空字符串、`nan` 混用。

### 1.3 序列层问题

- item 与 answer 版本不一致：同一被试可能出现“刺激序列版本”和“正确答案版本”不对应。
- 崩溃期参考被试替换错误：txt->xlsx 转换时引用错误模板，导致 item 序列错配。
- visit 子版本共存：除 visit1/2/3 外，还存在 `visit3_v1`、`visit3_v2`、`visit1_v1` 等任务级混合版本。

## 2. 修正原则

### 2.1 通用原则

- 所有修正先备份原始文件，再回写。
- item/answer 分组使用统一标准化：去路径、去扩展名、大小写统一、数字去前导 0、空值归一。
- 对齐判定以“任务级序列”而非仅数量统计。

### 2.2 关键规则

- `SST`：比较与分组只使用前 96 trial；超出部分截断。
- `EmotionStroop`：若目标 trial 缺失且其余序列匹配规则版本，则填充指定答案（如 `AN`）。
- `KT`：答案提取以 `answers[].answerPicName` 为准。
- `DCCS`：按已确认映射将数值答案映射至左右键语义。
- `ZYST`：比较使用 `ZYST_Formal` 口径。
- NBack：前导 `None`/空视为等价缺失，不作为版本冲突证据。

## 3. 本次（2026-02-13）执行步骤

### 3.1 修复 `answer_003` 的 `EmotionStroop` 缺失答案

- 对象：`groups_by_answer_regen_20260213/group_003_sublist.txt` 中 105 名被试。
- 动作：将 `EmotionStroop` 第 12 个 trial 的空答案填充为 `AN`。
- 结果：105/105 成功。
- 备份：`data/raw/behavior_data/cibr_app_data/_backup_fill_emotionstroop_trial12_AN_20260213_153123/`
- 明细：`data/interim/behavioral_preprocess/grouping_audit_regen_20260213/fill_emotionstroop_trial12_AN_report.json`

### 3.2 覆盖重建 20260213 分组

- answer 分组：`data/interim/behavioral_preprocess/groups_by_answer_regen_20260213/`
- item 分组：`data/interim/behavioral_preprocess/groups_by_item_regen_20260213/`
- 审计目录：`data/interim/behavioral_preprocess/grouping_audit_regen_20260213/`

重建后：

- answer=6 组、item=6 组；
- 639 名被试在两侧全覆盖；
- item 与 answer 达到 1:1 组对应。

### 3.3 确认版本目录规范化与结构化导出

- 目录重命名：
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed`
- -> `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/confirmed`

- 在 `confirmed/` 下为每个版本创建子目录并导出 18 个 task CSV：
- 子目录：`visit1_confirmed/`、`visit2_confirmed/`、`visit3_confirmed/`、`visit3_v1_confirmed/`、`visit3_v2_confirmed/`、`visit1_v1_confirmed/`
- 每个 task CSV 仅两列：`正式阶段刺激图片/Item名`、`正式阶段正确答案`

- 导出清单：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/confirmed/version_task_csv_export_manifest.csv`

## 4. 当前 `subject_id` -> `referred_visit` 映射表

- 文件：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/confirmed/subject_referred_visit.csv`
- 列结构：`subject_id`,`referred_visit`
- 总行数：639
- 映射元数据：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/confirmed/subject_referred_visit_meta.json`

当前 answer 组到 referred_visit 的规则为：

- `group_001` -> `visit1_confirmed`
- `group_002` -> `visit3_confirmed`
- `group_003` -> `visit3_v1_confirmed`
- `group_004` -> `visit2_confirmed`
- `group_005` -> `visit3_v2_confirmed`
- `group_006` -> `visit1_v1_confirmed`

## 5. 复现命令

```powershell
python -m scripts.export_confirmed_visit_assets
```

该命令会完成：

- `conformed -> confirmed` 重命名（若尚未重命名）；
- `subject_referred_visit.csv` 生成；
- 各 confirmed 版本 18-task 两列表格导出；
- `confirmed/manifest.json` 中 `conformed` 命名清理。

