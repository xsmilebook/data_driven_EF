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

### 3.1 证据优先级与关键假设

为保证推断可解释且尽量贴近“被试实际运行时的序列”，本次实现遵循以下优先级与假设：

- **主证据（设备端）**：`正式阶段正确答案` 被视为来自设备本地 txt 导出（见第 1 节），因此用于 visit 推断的主证据；但允许存在少量试次设置错误。
- **辅证据（时间线先验）**：从文件名 `THU_YYYYMMDD_*` 推断被试日期，并按用户确认的时间线给出“弱先验”（仅作轻微加权，不覆盖序列匹配证据）。
- **工作簿导出来源判别**：使用 `空屏时长` 列是否在所有任务 sheet 中都存在来启发式判别 `txt_export` vs `web_export`（规则见 `docs/reports/app_data_format.md`）。该判别用于后续追溯“崩溃期/参照替换/后台覆盖”的证据链构建，不直接参与 visit 推断。

### 3.2 visit 序列库构建（从 app_sequence 解析）

visit 序列配置从 `data/raw/behavior_data/app_sequence/` 读取，加载规则如下：

- **文件类型**：支持 `.json` 与 `.txt`（txt 内容为 json）；对编码异常做容错读取。
- **任务名规范化**：对文件名与 sheet 名进行统一映射（例如 `EmotionStoop→EmotionStroop`、`SpatialNBack→Spatial2Back`、`LG→EmotionSwitch`、`STROOP→ColorStroop`），并去除 `Formal` 等后缀，以降低命名不规范带来的丢任务风险。
- **刺激字段解析（Item）**：同时支持从 `picData` 与 `*PicName` 等键读取刺激名；若为路径则取文件名并去扩展名；纯数字字符串进行“去先导 0”归一化（用于解决如 `000100000` 与 `100000` 的等价性）。
- **答案字段解析**：优先从 `buttonName`（或大小写变体）读取，并做同样的归一化处理。
- **visit2–visit4 缺失任务补齐**：由于 visit2–visit4 可能存在任务配置缺失，本次以 visit1 为基线，将每个 visit 缺失的任务用 visit1 对应任务补齐，形成“effective visit sequences”（并将该序列库写入每次 run 的 `sequence_library/effective_visits_sequences.json`）。

### 3.3 工作簿观测序列提取（从 Excel 读取）

对每个被试工作簿（`*_GameData.xlsx`），从每个 sheet 提取两条观测序列：

- `正式阶段刺激图片/Item名`（来自网站后台导出，可能被覆盖，不作为 visit 推断主证据）
- `正式阶段正确答案`（来自设备端导出，作为 visit 推断主证据）

列定位策略为“表头精确匹配 + 关键词回退匹配”，并对单元格值进行归一化（去空白、路径裁剪、去扩展名、数字去先导 0）。

**SST 97 行异常处理**：当检测到 SST sheet 为 97 行（多出 1 行无效行）时，提取与写回前均会丢弃最后一行（详见 `docs/reports/app_data_format.md`）。

### 3.4 导出来源判别（txt_export vs web_export）

为支持“网站崩溃期/参照替换期”的后续定位，本次为每个工作簿输出一个导出来源的启发式判别：

- 若 `空屏时长` 列在该工作簿所有任务 sheet 中都存在，则标记为 `txt_export`；否则标记为 `web_export`。
- 该结论及证据字段（相关 sheet 数、缺失列的 sheet 列表等）写入每位被试的 `decisions/<subject_id>.json`，并在 `manifest.csv` 提供简化字段用于筛选。

### 3.5 visit 推断（答案序列匹配 + 弱先验）

visit 推断以“答案序列匹配”为核心：对每个候选 visit（visit1/2/3/4），计算该被试所有可比任务的答案匹配得分并取平均：

- **任务级匹配得分**：对齐到共同长度 `n=min(len(obs),len(exp))`，计算逐位相等比例，并对长度不一致做轻微惩罚（鼓励长度一致）。
- **跨任务汇总**：对该被试所有同时具备“观测答案序列 + 该 visit 的配置答案序列”的任务取平均作为该 visit 的总分。
- **日期弱先验**：若某个 visit 与时间线先验一致，则对该 visit 的总分增加一个很小的常数（仅用于打破近似平分，不覆盖序列证据）。

最终选择总分最高的 visit 作为 `inferred_visit`。

### 3.6 序列更正写回与输出

在推断得到 `inferred_visit` 后，将该 visit 的配置序列写回至工作簿对应列：

- 将 `正式阶段刺激图片/Item名` 与 `正式阶段正确答案` 两列分别替换为“推断 visit 的配置序列”。
- 其余列（按键、时间等）保持原样不动。
- 使用 `openpyxl` 直接修改并保存为新文件，从而尽量保持原始工作簿的 sheet、列顺序与格式；输出目录为新的 run 目录，不覆盖原始工作簿。

### 3.7 任务级异常标签（用于追溯）

为支持后续人工复核与问题归因，本次对每个“被试×任务”输出一个启发式 `issue_type`：

- `item_mismatch_suspected`：答案与推断 visit 高一致，但刺激列与推断 visit 明显不一致（可能为刺激列缺失/后台覆盖/参照替换等）。
- `backend_item_overwritten_suspected`：刺激列与某个其他 visit 高一致、但与推断 visit 不一致（更接近“网站当前序列覆盖”模式）。
- `answer_version_inconsistent`：答案序列更接近其他 visit（可能为设备版本差异或少量配置错误聚合的结果）。

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

### 5.1 手动剔除重复记录（与 `_excluded_duplicates` 对齐）

为与 `data/raw/behavior_data/cibr_app_data/_excluded_duplicates/` 的去重结果保持一致，已在 `run_corrected_v1` 中移除以下重复被试的更正输出（含 `subjects/`、`decisions/` 与 `manifest.*` 对应条目）：

- `THU_20231217_144_LSC_李斯晨_GameData.xlsx`
- `THU_20231230_161_ZJY_张佳茵_GameData.xlsx`
- `THU_20240120_183_DKZ_邓楷哲_GameData.xlsx`
- `THU_20241130_435_CJT_陈嘉桐_GameData.xlsx`
- `THU_20250605_601_ZYY_朱洋仪_GameData.xlsx`
- `THU_20250613_602_ZWZ_张威志_GameData.xlsx`
- `THU_20250709_628_ZSY_张烁杨_GameData.xlsx`
- `THU_20250725_684_LC_刘元博_GameData.xlsx`（保留 `THU_20250725_684_LC_李晨_GameData.xlsx`）

## 6. 局限与下一步（正式清洗脚本）

- visit3 的“子版本”（2025/07/09、2025/07/17、2025/07/29）目前仅通过答案匹配进行间接吸收，尚未构建任务级的细粒度版本库；后续需要在 `data/raw/behavior_data/app_sequence/` 上补齐子版本或从数据中反推子版本并固化。
- **关于“visit3 子版本是否仅答案不同、刺激不变”**：当前仓库的 `app_sequence/visit3/` 仅提供每个任务一份配置（未包含 07/09、07/17、07/29 的快照），因此无法仅凭配置文件严格证明“刺激是否完全不变”。从时间线记录看，至少存在“只改答案”的情形（2025/07/29 的 Emotion1Back），但也存在“任务被修改”的记录（2025/07/09 的 FZSS、2025/07/17 的 NBack），在缺少快照时不宜先验假设其刺激一定不变。后续建议：基于 visit3 时间窗内的设备端答案序列对每个任务进行聚类，形成子版本候选，并将子版本序列固化为可审计的版本库。
- “工作簿是否由 txt 导出生成”的判别仍属于启发式，但已加入一个强特征并输出可审计证据：若 `空屏时长` 列在所有任务 sheet 中都存在，则判定为 `txt_export`；否则判定为 `web_export`（证据字段写入 `decisions/*.json` 与 `manifest.csv`）。后续仍可叠加更多格式特征以提高鲁棒性与可解释性。
- 建议将本次推断逻辑固化为正式的统一清洗脚本（放入 `src/behavioral_preprocess/app_data/`，并提供稳定 CLI 入口），同时保留人工复核入口（输出疑似异常的被试与任务列表）。

### 6.1 v2 清洗中 item 的任务特例（KT / ZYST / SST）

在后续 v2 清洗中，item 侧证据不再以“与 visit 配置硬匹配”为唯一标准，而是优先使用“观测到的 item 类型（分组）+ 日期顺序”逐个核查。对于以下三个任务，按人工确认与字段结构，采用特例规则：

- `KT` / `ZYST`：任务序列在项目期间未更换，因此 visit1–visit4 的 **正确 item 均以 item group_001 模板为准**（`run_corrected_v2/sequence_library/visit1_item_group1_templates.json`）。后续做 item 类型比较时 **不纳入** `KT` / `ZYST`。
- `SST`：由于系统问题，工作簿的 `正式阶段刺激图片/Item名` 基本不记录有效 item，因此 `SST` 的 **正确 item 视为“全 null”**（保持 group_001 模板即可）。但 `SST` 的答案序列会随 visit 更换，后续答案推断/核查时仍需纳入 `SST`（仅不把 item 当作证据）。
