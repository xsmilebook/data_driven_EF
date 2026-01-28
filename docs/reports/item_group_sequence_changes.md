# item 分组（group_001–group_004）刺激序列差异分析（18 任务代表被试）

本文档对 `data/interim/behavioral_preprocess/groups_by_item/` 中的 `group_001–group_004` 做刺激序列（item）差异分析，目标是明确：这些 item group 在各任务上的 item 序列差异主要集中在哪些任务、是否与日期窗一致、以及哪些差异可能由“缺任务/缺列/导出格式”而非真实序列更换导致。

数据来源与方法：
- 原始工作簿：`data/raw/behavior_data/cibr_app_data/THU_*_GameData.xlsx`
  - `subject_id` 与工作簿文件名按前缀匹配：`{subject_id}_*_GameData.xlsx`
- 分组文件：`data/interim/behavioral_preprocess/groups_by_item/group_*_sublist.txt`
- item 提取列：`正式阶段刺激图片/Item名`
- 分组机制：`temp/group_app_stimulus_groups.py --grouping item`
  - 按 sheet 级 item 序列指纹分组；
  - 允许工作簿缺失部分 sheet，因此分组一致性只对“重叠 sheet”有约束。
- 代表被试选择规则（本次按你的要求更新）：
  - 在每个 group 内优先选择**完成 18 个任务（工作簿 sheet 数=18）**的被试；
  - 若存在多个候选，取日期最早者作为代表被试。
- 对比口径：
  - 对每个任务提取 item 序列并做基本清洗（去掉空值/None），对齐到共同长度后逐位比较得到 `match_ratio`；
  - `match_ratio` 用于定位差异任务，不等价于“全组一致性证明”（但足以快速定位“哪些任务在组间明显不同”）。

注意：
- 本次对比中 **KT / ZYST / SST 也纳入对比**（按你的最新要求）。
- `SST` 的 item 在工作簿中通常为大量缺失（清洗后近似空序列），因此在组间对比中常表现为“相同”（两边均为空）。但这不表示 SST 的**答案**一致；SST 仍需在答案侧推断中保留。
- `KT` 的 item 列在部分工作簿中仅出现**极少量非空记录**（例如只出现 1 个条目），因此其 `match_ratio` 更容易被“导出/记录不完整”放大；但目前已观测到不同 item（如 `Transport_1` vs `Transport_2`），提示其确实可能存在版本差异或被覆盖，因此仍需纳入考虑（建议与日期窗/答案侧证据联合判断）。

## 1. 各 item group 概览

| item_group | n_subjects | date_min | date_max | n_subjects_with_18_sheets | representative_subject_id | rep_tasks_n |
| --- | --- | --- | --- | --- | --- |
| group_001 | 391 | 2023-10-14 | 2025-02-14 | 347 | THU_20231203_144_LSC | 18 |
| group_002 | 229 | 2025-02-14 | 2025-11-22 | 215 | THU_20250214_570_WQX | 18 |
| group_003 | 16 | 2025-11-29 | 2025-12-20 | 11 | THU_20251129_801_LXY | 18 |
| group_004 | 2 | 2024-07-26 | 2025-09-21 | 0 | 无（该组没有 18-sheet 被试） | - |

备注：
- `group_004` 无法选取 18-sheet 代表被试，提示该组很可能与“缺任务/缺列/导出不完整”强相关；后续若要解释其 item 差异，应优先做格式诊断。

## 2. 以 group_001 为基准的 item 差异（包含 KT/ZYST/SST）

### 2.1 group_001 vs group_002

代表被试：group_001=`THU_20231203_144_LSC`，group_002=`THU_20250214_570_WQX`

差异任务（代表被试，match_ratio 越低表示 item 序列差异越大）：

| task | match_ratio | len_group_001 | len_group_002 |
| --- | --- | --- | --- |
| DT | 0.000 | 128 | 128 |
| FZSS | 0.000 | 120 | 120 |
| EmotionStroop | 0.010 | 96 | 96 |
| ColorStroop | 0.052 | 96 | 96 |
| CPT | 0.108 | 120 | 120 |
| Emotion2Back | 0.117 | 60 | 60 |
| Emotion1Back | 0.150 | 60 | 60 |
| Spatial2Back | 0.083 | 60 | 60 |
| Spatial1Back | 0.217 | 60 | 60 |
| DCCS | 0.225 | 40 | 40 |
| KT | 0.000 | 1 | 1 |

完全一致的任务（代表被试层面）：
- `EmotionSwitch`、`FLANKER`、`GNG`、`Number1Back`、`Number2Back`、`SST`、`ZYST`

解释要点：
- `DT`/`FZSS` 在代表被试上呈现“完全不同”的 item 序列，符合“刺激序列版本切换/网站端覆盖”导致的强差异模式。
- `Emotion1Back/Emotion2Back` 在当前 18-sheet 代表被试中可直接比较，其 item 序列也呈现明显差异（match_ratio 约 0.12–0.15），提示该差异更可能是“真实版本差异/覆盖”而非早期缺失导致的假差异。
- `KT` 的差异来自 item 列仅出现极少数非空记录（例如在两位代表被试中分别只出现 `Transport_1` vs `Transport_2`），更可能反映记录稀疏/导出差异，而不是稳定的序列版本切换信号。

### 2.2 group_001 vs group_003

代表被试：group_001=`THU_20231203_144_LSC`，group_003=`THU_20251129_801_LXY`

差异任务：

| task | match_ratio | len_group_001 | len_group_003 |
| --- | --- | --- | --- |
| EmotionStroop | 0.010 | 96 | 96 |
| Emotion2Back | 0.117 | 60 | 60 |
| Emotion1Back | 0.150 | 60 | 60 |

解释要点：
- 相比 group_002，group_003 与 group_001 的差异集中在 `Emotion1Back/Emotion2Back/EmotionStroop`；其中 `EmotionStroop` 的差异幅度很小（96 位中约 1 位不同），更像“单 trial 异常/导出噪声/轻微命名差异”，但 NBack 的差异更明显（match_ratio 约 0.12–0.15），需要进一步结合日期窗与设备信息定位其来源。
- 代表被试层面，group_001 与 group_003 在多数任务上 item 完全一致（如 `CPT/ColorStroop/DCCS/DT/FZSS/Spatial1Back/Spatial2Back` 等），提示这两个组可能仅在部分任务受到覆盖/版本差异影响。

### 2.3 group_001 vs group_004

该组没有 18-sheet 被试，因此本次不做代表被试对比；建议优先从“缺任务/缺列/导出不完整”角度诊断。

## 3. 结论与后续建议

初步结论（代表被试层面）：
- item group 的主要强差异出现在 `DT` 与 `FZSS`（group_001 vs group_002：match_ratio=0），提示这些任务最可能存在明确的 item 版本差异或覆盖机制。
- `Emotion1Back/Emotion2Back` 也在 group_001 vs group_002/003 中呈现明显差异（match_ratio 约 0.12–0.15），提示其 item 可能存在版本差异或覆盖机制。
- `EmotionStroop` 在 group_001 vs group_002/003 均呈现极小差异；需要进一步核查该差异是否稳定、是否只集中在某一个 trial。
- `group_004` 更像“缺任务/缺列导致无法并入主组”的边缘组，应单独做格式诊断。

建议下一步：
1. 对每个 group 在每个关键任务上各抽取多个代表被试（例如 3–5 名，覆盖日期窗口两端），检验差异是否稳定。
2. 对 `group_004` 优先做格式诊断（缺失 sheet、缺失列、异常行数等），避免把“导出不完整”误判为“新序列版本”。
3. 将 item 差异与日期时间线联动：在 2025/07/08（visit3）、2025/11/28（visit2）、2025/12/16（回到 visit1）附近重点检查 `DT/FZSS/Emotion*Back` 是否出现一致的切换点。
