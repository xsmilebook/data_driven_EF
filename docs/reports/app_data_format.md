# App 行为数据格式整理（CIBR GameData）

本文档记录 `data/raw/behavior_data/cibr_app_data/` 下单被试 Excel 工作簿（`*_GameData.xlsx`）的关键格式约定、已观测到的异常情况，以及当前临时处理策略。本文档仅用于支撑临时分析与分组；后续将基于异常汇总编写正式、统一的清洗脚本（并纳入可复现流水线）。

## 1. 文件与工作簿结构

- 输入形式：每位被试一个 Excel 工作簿（`*.xlsx`），多个任务对应多个 sheet。
- 被试标识：文件名包含被试编码（例如 `THU_YYYYMMDD_<id>_<code>_*_GameData.xlsx`），脚本默认用前 4 段下划线字段拼接为 `subject_id`（例：`THU_20251207_808_QYH`）。
- 关键列（不同任务 sheet 通常共享）：
  - `任务`
  - `正式阶段刺激图片/Item名`
  - `正式阶段正确答案`
  - `正式阶段被试按键`
  - 其他时间与汇总列（如 `相对时间(秒)`、`平均反应时(秒)`、`正确率`）

## 2. 已观测异常：SST sheet 存在 97 行

### 2.1 异常描述

在 SST 任务中，部分被试的 `SST` sheet 行数为 **97 行**（相对常见的 96 行多出 1 行）。该额外的最后一行不属于有效试次记录，可视为导出/拼接过程引入的无效行。

该异常会影响以“序列一致性”为依据的跨被试分组或对齐逻辑，例如：

- 按 `正式阶段正确答案` 的答案序列比较时，多出的 1 行会导致长度不一致，从而被错误判定为不同组。

### 2.2 当前临时处理策略（用于分组比较）

在现阶段的分组（`temp/group_app_stimulus_groups.py`）中（无论按 `正式阶段刺激图片/Item名` 还是按 `正式阶段正确答案`）：

- 当某个 sheet 满足：
  - 存在 `任务` 列，且 `len(df) == 97`
  - `任务` 列的首行值为 `SST`
- 则在生成分组比较序列前，**丢弃最后一行**（`df = df.iloc[:-1]`），使 SST 的比较仅基于有效行。

该处理仅用于分组对比，不代表最终清洗口径。

## 2.3 当前分组输出位置

- 按 `正式阶段正确答案`：`data/interim/behavioral_preprocess/groups_by_answer/`
- 按 `正式阶段刺激图片/Item名`：`data/interim/behavioral_preprocess/groups_by_item/`

## 2.4 序列更正输出位置

若需要基于 visit 序列配置对“刺激/答案序列”进行一致化更正（不覆盖原始工作簿），更正版工作簿输出到：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/`（run 目录结构详见 `docs/workflow.md`）

更正过程与全样本统计见：`docs/reports/app_sequence_correction_report.md`。

## 3. 判断工作簿是否由本地 txt 导出（临时规则）

基于全样本的格式差异观察，可以使用如下**强判别特征**来区分“由设备本地 txt 导出生成的 Excel”与“由网站后台导出的 Excel”：

- 若 `空屏时长` 列在该工作簿的**所有任务 sheet**中都存在（以表头列名为准），则可判定该工作簿更可能来自**本地 txt 导出**。
- 若 `空屏时长` 列并非在所有任务 sheet 中都存在，则更可能来自**网站后台导出**（网站导出通常只在少数任务/少数 sheet 中包含该列）。

该规则已被纳入更正脚本的可审计输出：

- `python -m scripts.correct_app_workbooks ...` 生成的 `decisions/<subject_id>.json` 中会写入 `export_source` 及其证据字段（`blank_screen_in_all_sheets` 等）。
- 同时在 `manifest.csv` 中写入 `export_source` 与 `blank_screen_in_all_sheets`，便于后续统计与筛选。

注意：该规则用于**判断导出来源**，不直接等价于“刺激列是否被后台覆盖/参照替换”。导出来源判别只是后续推断“崩溃期/参照替换期”的一条关键证据。

## 4. 后续工作（计划）

- 基于当前分组过程与对比输出，系统化汇总所有结构异常（包括但不限于：行数异常、列名不一致、空行/重复行、关键列缺失）。
- 在 `src/behavioral_preprocess/` 下实现正式的、可复现的统一清洗脚本，并将该异常处理（如 SST 97 行）纳入规范化步骤。
