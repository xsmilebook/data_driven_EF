# APP data cleaning v2 workflow (draft)

本文档记录基于 `data/raw/behavior_data/cibr_app_data/` 生成新版 APP 行为数据清洗结果的建议流程（v2）。目标是在保留可审计证据链的前提下，按日期顺序、结合 item/answer 的“类型（序列指纹）”逐个排查并修正 `docs/notes/app_data_problem.md` 中描述的问题。

## 0. 为什么需要 v2（v1 的主要漏洞）

v1 方案以“将被试答案序列与 visit 配置做匹配”作为核心证据，但存在结构性漏洞：

1. **visit 配置本身不完整**：在 `app_sequence` 里部分任务的 `answers`（甚至 `items`）无法从配置文件中正确解析，导致期望序列包含大量 `None`，使匹配分数失真。
   - 例：visit1 中 `CPT.answers` 与 `GNG.answers` 全为 `None`；`KT.items/answers` 与 `ZYST.items/answers` 全为 `None`。
   - 直接后果：这些任务会系统性产生“与 visit1 不一致”的假阳性或不可解释的低分。
2. **导出链路导致 item/answer 来源不同**：答案更接近设备端真实序列，item 可能被网站覆盖或被参照被试替换；对单一 visit 的硬匹配无法区分“真实版本差异”与“字段被覆盖”。
3. **visit3 子版本未建库**：2025/07/09、07/17、07/29 的子版本差异若不显式建库，仅靠整体匹配会将多版本混在一起。

因此 v2 需要把“匹配 visit 配置”降级为辅助信息，把核心证据改为“按日期顺序 + 观测到的 item/answer 序列类型（聚类/指纹）”。

## 1. v2 总体策略（按日期顺序 + 序列类型）

### 1.1 基本单位

- 单位：`subject_id × task × {item, answer}`。
- 对每个 sheet 的 item 或 answer 提取“观测序列”（做统一归一化：去路径、去扩展名、数字去先导 0、空值标准化等）。
- 为每条观测序列计算一个**稳定指纹**（例如 sha256(hash(json(items)))），形成 `task × field` 维度的“类型 ID”。

### 1.2 按日期排序的排查流程

1. 将所有被试按 `THU_YYYYMMDD_*` 的日期升序排序。
2. 对每个任务分别查看：在时间线上出现了哪些 `item_type` 与 `answer_type`，以及它们的切换点是否与已知时间线一致。
3. 在每个切换窗口中重点排查：
   - **设备未更新**：日期已经进入新 visit，但 `answer_type` 仍属于旧类型。
   - **网站覆盖**：`item_type` 与同日/同批次多数人不一致，或与“网站当前序列”高度一致。
   - **崩溃期参照替换**：多个被试的 `item_type` 异常一致且与其 `answer_type`、日期先验明显冲突。
4. 对于每个疑点被试，逐任务做“字段级修正决定”，并写入可审计的 decision log。

## 2. 现有分组产物的检查结果（data/interim）

当前 `data/interim/behavioral_preprocess/` 下已有按序列指纹分组的结果（由 `temp/group_app_stimulus_groups.py` 生成）。在 v2 中仍可作为探索性工具，但需要先确认它们是否与当前 raw 数据一致。

检查脚本：

- `python -m scripts.audit_app_groupings --dataset EFNY_THU --config configs/paths.yaml --write-report`
- 报告输出：`docs/reports/app_grouping_audit.md`

核心结论（以当前 raw 目录重新生成分组后）：

- **answer 分组类型数：9**；**item 分组类型数：4**（符合“刺激版本更少、答案版本更多”的预期）。
- 分组文件与当前 raw 被试完全一致（`missing_in_raw=0`），且无重复行（`duplicate_lines=0`）。

## 2.1 v2 的 visit1 序列库构建（先用 item group_001 + app_sequence/visit1）

第一步可以先把 visit1 的“可写回模板”建立起来，后续再扩展到 visit2/3/4 以及子版本：

- item：以 `data/interim/behavioral_preprocess/groups_by_item/group_001_sublist.txt`（item group_001）作为 visit1 的刺激序列参照，读取代表被试工作簿，提取每个任务的 `正式阶段刺激图片/Item名` 列作为“列模板”（保留 None 的位置模式，保证能写回到 Excel）。  
- answer：从 `data/raw/behavior_data/app_sequence/visit1/` 解析得到 visit1 的答案序列；对 CPT/GNG 等任务，使用 `needClickButton` 作为答案（与工作簿中的布尔值一致）；对 KT/ZYST 使用 `answerPicName` 作为答案序列。

生成脚本：

- `python -m scripts.build_visit1_library_v2 --dataset EFNY_THU --config configs/paths.yaml`

输出位置（v2 run 目录，仅序列库，不生成更正后的 Excel）：  

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit1_item_group1_templates.json`  
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit1_from_app_sequence.json`  
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/sequence_library/visit1_merged.json`

## 2.2 item 比对的任务特例（KT / ZYST / SST）

根据人工确认与数据结构特点，v2 中对 item 的处理采用以下特例规则（适用于 visit1–visit4 的 item 侧判断）：

- `KT` / `ZYST`：任务序列在项目期间未更换，因此 **item 永远以** `visit1_item_group1_templates.json` 中的模板为准即可；在“比对 item 的版本差异”步骤中可 **不纳入** `KT` / `ZYST`（避免被其字段结构影响）。
- `SST`：由于系统问题，工作簿的 `正式阶段刺激图片/Item名` 基本不记录有效 item；因此 **SST item 保持模板中的全 null 即可**。但 `SST` 的 **answer 会随 visit 更换**，后续在答案类型推断/核查时需要继续纳入 `SST`（只是不把 item 当作证据）。

## 3. v2 的可复现输出与记录规范（建议）

### 3.1 输出目录

- 建议输出新 run：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/`（不覆盖 v1）。

### 3.2 需要写入的审计文件

- `sequence_catalog.json`：记录每个 `task×field` 下的全部类型（type_id、长度、代表被试、日期范围、与已知 visit 的关联假设）。
- `decisions/<subject_id>.json`：逐被试记录每个任务的 `item_type`/`answer_type`、修正动作、证据（日期窗口、同组被试、与其他类型差异等）。
- `manifest.csv/json`：汇总层面（每被试整体标签、冲突数量、需要人工复核标记等）。

## 4. 下一步实现要点（工程层面，待实现）

- 先修复 `app_sequence` 配置解析：补齐任务的 item/answer 抽取规则（避免 `None` 序列导致的伪差异）。
- 实现“按日期排序 + 类型 catalog”的扫描器，产出可审计的类型清单与疑点列表。
- 在不修改 `data/raw` 的前提下生成 v2 更正结果，并与 v1 做回归对比（分组数、冲突率、疑点数量）。
