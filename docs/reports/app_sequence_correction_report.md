# APP 行为数据序列推断与更正报告（EFNY）

本报告面向 `data/raw/behavior_data/cibr_app_data/` 下的 APP 行为工作簿（`*_GameData.xlsx`），在不覆盖原始文件的前提下，结合 `data/raw/behavior_data/app_sequence/` 提供的 visit1–visit4 序列配置，对“正式阶段刺激图片/Item名”与“正式阶段正确答案”进行版本推断与一致化更正，并给出全样本异常统计。

## 1. 背景：导出链路导致的序列不一致

APP 数据导出存在“同一份工作簿内，不同字段来源不同”的结构性问题：

1) **设备端（本地 txt 导出）字段**：`正式阶段正确答案`、`正式阶段被试按键`、`相对时间(秒)` 等由执行任务的设备本地 txt（json 格式）导出，理论上反映被试实际运行时所用序列。
2) **网站后台字段**：`正式阶段刺激图片/Item名` 由网站后台导出，可能被“当前网站存储的序列”覆盖，而非被试实际运行时使用的序列。
3) **序列多次更换**：采集过程中存在多次更换/修订（含局部任务修订），导致同一时间窗内可能存在多版本共存；且存在“正确答案配置错误”的可能，使得答案序列并非 100% 可靠。
4) **网站崩溃期的参照替换**：崩溃期间部分数据通过本地 txt 导出，但刺激列可能引用某个“参照被试”的刺激序列，导致刺激序列与答案/按键等字段脱钩。

## 2. visit 序列配置与时间线先验

### 2.1 配置来源与不完整性

序列配置位于 `data/raw/behavior_data/app_sequence/`，包含 visit1、visit2（目录名可能为 `visit2-*`）、visit3、visit4：

- 配置格式：json 或 txt（其中 txt 内容为 json）。
- **已知问题**：visit2–visit4 的任务配置可能不完整；部分文件命名不规范（如 `EmotionStoop_Formal`、`SpatialNBack_Formal`）。
- 处理策略：以 visit1 为基线，对 visit2/3/4 缺失任务用 visit1 对应任务配置补齐（假设“未出现配置文件的任务未改动”）。

### 2.2 序列更换时间（用户确认）

- 项目开始阶段：visit1
- 2025/07/08：更换为 visit3
- 2025/07/09：修改 visit3 的 FZSS
- 2025/07/17：修改 visit3 的 spatial-nback 与 emotion-nback
- 2025/07/29：修改 visit3 的 Emotion1Back 答案
- 2025/11/28：更换为 visit2
- 2025/12/16：更换为 visit1
- 2026/01/13：更换为 visit4

## 3. 推断与更正方法（本次实现）

实现脚本：`python -m scripts.correct_app_workbooks --dataset EFNY --config configs/paths.yaml`

核心策略：

- **推断依据**：以 `正式阶段正确答案` 为主进行 visit 推断（因为来自设备端 txt），并使用被试日期（从 `THU_YYYYMMDD_*` 推断）作为弱先验。
- **更正目标**：将每个被试工作簿中的
  - `正式阶段刺激图片/Item名`
  - `正式阶段正确答案`
  更正为“推断 visit 对应的序列配置”。
- **SST 97 行异常**：若 SST sheet 为 97 行（最后一行无效），更正前会删除最后一行后再进行写入与比较（详见 `docs/reports/app_data_format.md`）。

输出为“每被试一个更正后的 Excel”，并保留其他列不变。

## 4. 全样本分析结果（run_corrected_v1）

本次处理的工作簿数量：646。

### 4.1 被试 visit 推断分布

推断结果（基于答案序列匹配 + 日期先验）：

- visit1：457
- visit2：15
- visit3：174
- visit4：0（当前样本最晚到 2025/12/20）

### 4.2 异常类型统计（任务级）

基于“推断 visit 的配置序列”与原始工作簿序列的匹配情况，定义并统计以下任务级异常：

- `item_mismatch_suspected`：答案与推断 visit 高一致，但刺激列与推断 visit 明显不一致（疑似后台覆盖/参照替换/缺失）。
- `backend_item_overwritten_suspected`：刺激列与某个其他 visit 高一致、但与推断 visit 不一致（更接近“网站当前序列覆盖”模式）。
- `answer_version_inconsistent`：答案序列更接近其他 visit（可能为设备版本差异或少量配置错误聚合的结果）。

统计结果（以“任务×被试”为单位）：

- `item_mismatch_suspected`：780
- `answer_version_inconsistent`：212
- `backend_item_overwritten_suspected`：196

若按“涉及该异常的被试数”统计（以被试为单位）：

- `item_mismatch_suspected`：458
- `answer_version_inconsistent`：178
- `backend_item_overwritten_suspected`：70

高频任务（`item_mismatch_suspected`）：

- SST：455（该任务刺激列常见为空，导致与配置序列不一致）
- Emotion1Back：54
- Emotion2Back：54
- Spatial1Back：54
- Spatial2Back：54

高频任务（`backend_item_overwritten_suspected`，更接近“网站当前序列覆盖”模式）：

- FZSS：68
- DT：67
- Number1Back：15
- Number2Back：14

## 5. 更正后数据集（不覆盖原始）

输出目录由 `configs/paths.yaml` 控制（`dataset.behavioral.corrected_app_excel_dir`），本次 run 为：

- 更正后的 Excel：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v1/subjects/`
- 总清单：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v1/manifest.csv`
- 逐被试推断与任务级诊断：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v1/decisions/<subject_id>.json`
- 本次使用的“补齐后的 visit 序列库”：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v1/sequence_library/effective_visits_sequences.json`

其中 `manifest.csv` 与 `decisions/*.json` 额外包含“导出来源（txt 导出 vs 网站导出）”的启发式判别字段（见 `docs/reports/app_data_format.md`）。

## 6. 局限与下一步（正式清洗脚本）

- visit3 的“子版本”（2025/07/09、2025/07/17、2025/07/29）目前仅通过答案匹配进行间接吸收，尚未构建任务级的细粒度版本库；后续需要在 `data/raw/behavior_data/app_sequence/` 上补齐子版本或从数据中反推子版本并固化。
- “工作簿是否由 txt 导出生成”的判别仍属于启发式，但已加入一个强特征并输出可审计证据：若 `空屏时长` 列在所有任务 sheet 中都存在，则判定为 `txt_export`；否则判定为 `web_export`（证据字段写入 `decisions/*.json` 与 `manifest.csv`）。后续仍可叠加更多格式特征以提高鲁棒性与可解释性。
- 建议将本次推断逻辑固化为正式的统一清洗脚本（放入 `src/behavioral_preprocess/app_data/`，并提供稳定 CLI 入口），同时保留人工复核入口（输出疑似异常的被试与任务列表）。
