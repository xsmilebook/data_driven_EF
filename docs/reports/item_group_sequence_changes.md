# item 分组（group_001–group_004）刺激序列差异分析

本文档对 `data/interim/behavioral_preprocess/groups_by_item/` 中的 `group_001–group_004` 做刺激序列（item）差异分析，目标是明确：这些 item group 在各任务上的 item 序列差异主要集中在哪些任务、是否与日期窗一致、以及哪些差异可能由“缺任务/缺列/导出格式”而非真实序列更换导致。

数据来源与方法：
- 原始工作簿：`data/raw/behavior_data/cibr_app_data/*.xlsx`
- 分组文件：`data/interim/behavioral_preprocess/groups_by_item/group_*_sublist.txt`
- item 提取列：`正式阶段刺激图片/Item名`
- 分组机制：`temp/group_app_stimulus_groups.py --grouping item`（按 sheet 级 item 序列指纹分组；允许工作簿缺失部分 sheet，因此分组一致性只对“重叠 sheet”有约束）
- 本文中的“match_ratio”以每个 group 的**代表被试**（该组最早日期被试）提取出的观测 item 序列为基础，对齐到共同长度后逐位比较得到；其用途是定位差异任务，而非严格的全组一致性证明。

重要约束（v2 清洗规则相关）：
- `KT`/`ZYST`：序列不更换，item 不用于版本差异比较。
- `SST`：item 基本不记录，保持模板全 null；因此 **不使用 SST item** 判定 group 差异。

## 1. 各 item group 概览

| item_group | n_subjects | date_min | date_max | representative_subject_id | rep_tasks_n |
| --- | --- | --- | --- | --- | --- |
| group_001 | 391 | 2023-10-14 | 2025-02-14 | THU_20231014_131_ZXM | 16 |
| group_002 | 229 | 2025-02-14 | 2025-11-22 | THU_20250214_570_WQX | 18 |
| group_003 | 16 | 2025-11-29 | 2025-12-20 | THU_20251129_801_LXY | 18 |
| group_004 | 2 | 2024-07-26 | 2025-09-21 | THU_20240726_337_DYN | 6 |

备注：
- `group_004` 的代表被试仅包含 6 个任务 sheet（明显少于常规 16–18 个任务），提示该 group 可能主要由“工作簿缺任务/缺列”导致无法与其他组建立充分重叠约束，从而被分离出来；对其差异解释需谨慎。

## 2. 以 group_001 为基准的 item 差异（排除 KT/ZYST/SST）

### 2.1 group_001 vs group_002

代表被试：group_001=`THU_20231014_131_ZXM`，group_002=`THU_20250214_570_WQX`

差异任务（代表被试，match_ratio 越低表示 item 序列差异越大）：

| task | match_ratio | len_group_001 | len_group_002 |
| --- | --- | --- | --- |
| DT | 0.000 | 128 | 128 |
| FZSS | 0.000 | 120 | 120 |
| EmotionStroop | 0.010 | 96 | 96 |
| ColorStroop | 0.052 | 96 | 96 |
| CPT | 0.108 | 120 | 120 |
| Spatial2Back | 0.083 | 60 | 60 |
| Spatial1Back | 0.217 | 60 | 60 |
| DCCS | 0.225 | 40 | 40 |

代表被试中“仅在 group_002 出现而 group_001 缺失”的任务：
- `Emotion1Back`、`Emotion2Back`

解释要点：
- `DT`/`FZSS` 在代表被试上呈现“完全不同”的 item 序列，符合“刺激序列版本切换/网站端覆盖”导致的强差异模式。
- `Emotion1Back/Emotion2Back` 在 2023–2024 的早期工作簿中可能缺失或未导出（至少在 group_001 代表被试中缺失），因此不能直接用“是否出现”判断其是否更换过序列，应进一步以更晚日期且仍属 group_001 的被试做补充核查。

### 2.2 group_001 vs group_003

代表被试：group_001=`THU_20231014_131_ZXM`，group_003=`THU_20251129_801_LXY`

差异任务：

| task | match_ratio | len_group_001 | len_group_003 |
| --- | --- | --- | --- |
| EmotionStroop | 0.010 | 96 | 96 |

代表被试中“仅在 group_003 出现而 group_001 缺失”的任务：
- `Emotion1Back`、`Emotion2Back`

解释要点：
- 相比 group_002，group_003 与 group_001 的 item 差异非常少，仅在 `EmotionStroop` 上出现极小幅度差异（代表被试 96 位中约 1 位不同）。这更像“单 trial 异常/导出噪声/轻微命名差异”而非大范围序列更换，但仍需结合更多代表被试复核。

### 2.3 group_001 vs group_004

代表被试：group_001=`THU_20231014_131_ZXM`，group_004=`THU_20240726_337_DYN`

差异任务（仅基于双方都存在的任务）：

| task | match_ratio | len_group_001 | len_group_004 |
| --- | --- | --- | --- |
| DT | 0.000 | 128 | 128 |
| Number1Back | 0.000 | 60 | 60 |
| Number2Back | 0.000 | 60 | 60 |

代表被试中“仅在 group_004 出现而 group_001 缺失”的任务：
- `Emotion1Back`、`Emotion2Back`

代表被试中“仅在 group_001 出现而 group_004 缺失”的任务（大量缺失）：
- `CPT`、`ColorStroop`、`DCCS`、`EmotionStroop`、`EmotionSwitch`、`FLANKER`、`FZSS`、`GNG`、`Spatial1Back`、`Spatial2Back`

解释要点：
- 由于 group_004 代表被试严重缺任务，分组约束较弱，其“差异”可能混合了真实序列差异与导出缺失导致的假分离；该组建议优先从“格式/缺任务”角度排查，而不是先验视为新序列版本。

## 3. 结论与后续建议

初步结论（代表被试层面）：
- item group 的主要强差异出现在 `DT` 与 `FZSS`（group_001 vs group_002：match_ratio=0），提示这些任务最可能存在明确的 item 版本差异或覆盖机制。
- `EmotionStroop` 在 group_001 vs group_003 也呈现极小差异；需要进一步核查该差异是否稳定、是否只集中在某一个 trial。
- `group_004` 更像“缺任务/缺列导致无法并入主组”的边缘组，应单独做格式诊断。

建议下一步：
1. 对每个 group 在每个关键任务上各抽取多个代表被试（例如 3–5 名，覆盖日期窗口两端），检验差异是否稳定。
2. 对 `Emotion1Back/Emotion2Back` 的“早期缺失”问题，明确是历史上未采集/未导出，还是脚本未能识别对应 sheet/列。
3. 在 v2 中 item 比对时继续排除 `KT`/`ZYST`/`SST`，并优先用 `DT/FZSS` 等信息量更高的任务做版本判定。

