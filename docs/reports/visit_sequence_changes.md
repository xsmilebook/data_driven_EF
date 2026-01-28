# APP visit 序列改动说明（visit1–visit4）

本文档汇总 `visit1/visit2/visit3/visit4` 的序列改动与时间线，供 APP 行为数据清洗与复核使用。

## 数据来源与约束

- 序列配置来源（历史参考）：`data/raw/behavior_data/app_sequence/`。
- 采用的有效序列库：`data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v1/sequence_library/effective_visits_sequences.json`。
- 约束：visit2–visit4 若缺失任务配置，会用 visit1 补齐（因此“未列出的改动”不等于“真实未改动”，仅表示配置中未观察到差异）。
- 重要说明：由于当前阶段认为 `app_sequence` 配置（尤其是答案序列）**不可靠**，本文件中的“差异概览”仅作为辅助线索与时间线记录；v2 清洗应以工作簿观测到的 `item/answer` 类型分组与日期顺序作为主证据。

## 时间线（确认记录）

- 项目开始阶段：visit1。
- 2025/07/08：切换到 visit3。
- 2025/07/09：visit3 的 FZSS 修改。
- 2025/07/17：visit3 的 Spatial/Emotion NBack 修改。
- 2025/07/29：visit3 的 Emotion1Back 答案修改。
- 2025/11/28：切换到 visit2。
- 2025/12/16：切换回 visit1。
- 2026/01/13：切换到 visit4。

## 与 visit1 的差异概览（基于 effective 序列库）

### visit2 相对 visit1

- 刺激序列（Item）不同：`CPT`、`ColorStroop`、`DCCS`、`DT`、`FLANKER`、`FZSS`、`Number1Back`、`Number2Back`、`SST`、`Spatial1Back`、`Spatial2Back`。
- 正确答案不同：`ColorStroop`、`DCCS`、`DT`、`FLANKER`、`FZSS`、`Number2Back`、`SST`、`Spatial1Back`、`Spatial2Back`。

### visit3 相对 visit1

- 刺激序列（Item）不同：`CPT`、`ColorStroop`、`DCCS`、`DT`、`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`FZSS`、`SST`、`Spatial1Back`、`Spatial2Back`。
- 正确答案不同：`ColorStroop`、`DCCS`、`DT`、`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`FZSS`、`SST`、`Spatial1Back`、`Spatial2Back`。

### visit4 相对 visit1

- 刺激序列（Item）不同：`CPT`、`ColorStroop`、`DCCS`、`DT`、`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`FZSS`、`SST`、`Spatial1Back`、`Spatial2Back`。
- 正确答案不同：`ColorStroop`、`DCCS`、`DT`、`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`FZSS`、`SST`、`Spatial1Back`、`Spatial2Back`。

## 与 visit1 完全一致的任务（Item + Answer 均一致）

- visit2 与 visit1 一致：`Emotion1Back`、`Emotion2Back`、`EmotionStroop`、`EmotionSwitch`、`GNG`、`ZYST`。
- visit3 与 visit1 一致：`EmotionSwitch`、`FLANKER`、`GNG`、`Number1Back`、`Number2Back`、`ZYST`。
- visit4 与 visit1 一致：`EmotionSwitch`、`FLANKER`、`GNG`、`Number1Back`、`Number2Back`、`ZYST`。

## 备注

- visit3 的子版本改动（2025/07/09、2025/07/17、2025/07/29）目前未形成独立配置快照，因此在 effective 序列库中不会体现任务级子版本差异；需要时应基于答案序列做子版本聚类或补齐配置快照。
- 刺激序列与正确答案的来源不同（设备端 vs 网站端）会导致“刺激/答案不一致”的现象，详见 `docs/reports/app_sequence_correction_report.md` 与 `docs/notes/app_data_problem.md`。
- `KT`：尽管在 effective 序列库（配置侧）中未观察到明确改动，但在工作簿观测到的 item 序列存在差异（且 item 分组对比中出现明显不同），因此在 v2 清洗中应将 `KT` 作为需要纳入 item 证据的任务，而不应按“永远一致”处理。
