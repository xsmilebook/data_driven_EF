# 行为数据探索性分析：试次数量与条件分布（单被试）

## 数据来源
- 工作簿：`D:/projects/data_driven_EF/data/raw/behavior_data/cibr_app_data/THU_20250829_759_JZC_景志成_GameData.xlsx`
- Sheet 数：18

## 口径说明
- 本报告用于探索性试次统计与建模可行性评估：**不进行反应时过滤/修剪**（避免单被试过滤后分布失真）。
- `Included` 表示可被当前规则归类的试次数量（例如切换任务需可判定 repeat/switch；SST 可能截断到前 96 行）。

## Go/NoGo（基于 answer 的 true/false）
| Task | Total | Go | NoGo | Note |
| --- | --- | --- | --- | --- |
| CPT | 120 | 36 | 84 |  |
| GNG | 100 | 80 | 20 |  |

## SST：Go/Stop（基于 SSD/SSRT 列是否为空）
| Task | Total | Included | Go | Stop | Note |
| --- | --- | --- | --- | --- | --- |
| SST | 97 | 96 | 72 | 24 | truncated_to_96 |

## Repeat/Switch（按规则变更；含 mixed_from 截断）
| Task | Total | Included | Repeat | Switch | Note |
| --- | --- | --- | --- | --- | --- |
| DT | 128 | 63 | 32 | 31 |  |
| EmotionSwitch | 128 | 63 | 38 | 25 |  |
| DCCS | 40 | 19 | 10 | 9 |  |

## Congruent/Incongruent（基于 item 解析）
| Task | Total | Included | Congruent | Incongruent | Note |
| --- | --- | --- | --- | --- | --- |
| EmotionStroop | 96 | 96 | 24 | 72 |  |
| FLANKER | 96 | 96 | 48 | 48 |  |
| ColorStroop | 96 | 96 | 24 | 72 |  |

## N-back（Match/NoMatch；基于 item 的 n-back lag）
| Task | Total | Included | Match | NoMatch | Note |
| --- | --- | --- | --- | --- | --- |
| oneback_spatial | 60 | 60 | 12 | 48 |  |
| twoback_spatial | 60 | 60 | 12 | 48 |  |
| oneback_number | 60 | 60 | 12 | 48 |  |
| twoback_number | 60 | 60 | 11 | 49 |  |
| oneback_emotion | 60 | 60 | 7 | 53 |  |
| twoback_emotion | 60 | 60 | 8 | 52 |  |

## 其他任务（仅总试次）
| Task | Total |
| --- | --- |
| FZSS | 120 |
| KT | 9 |
| ZYST | 128 |

## Drift model（DDM）适用性建议
以下判断以“二选一反应（2AFC）+ 试次级 RT + 充分试次数量”为基本前提；若任务存在抑制/停止机制或大量无反应试次，应优先考虑扩展模型（如 go/no-go DDM、stop-signal race）。

- **优先候选（冲突类）**：EmotionStroop, FLANKER, ColorStroop（条件划分清晰，通常可按 congruent/incongruent 分层拟合）。
- **可用候选（切换类）**：DT, EmotionSwitch（可按 repeat/switch 分层；需确保响应为二选一且 RT 质量可控）。
- **谨慎使用（切换类试次偏少）**：DCCS(Included=19)（分层后单条件试次可能不足，参数估计不稳定）。
- **可探索（N-back）**：oneback_spatial, twoback_spatial, oneback_number, twoback_number, oneback_emotion, twoback_emotion（可按 match/no-match 分层；需注意 match 试次往往较少，可能不足以稳定估计条件差异）。
  - 本工作簿中 match 试次偏少：oneback_spatial(Match=12), twoback_spatial(Match=12), oneback_number(Match=12), twoback_number(Match=11), oneback_emotion(Match=7), twoback_emotion(Match=8)。
- **不建议直接用标准 2AFC DDM（Go/NoGo）**：CPT, GNG（no-go 通常缺失 RT；若要建模可考虑 go/no-go DDM 或 race 模型）。
- **不建议直接用标准 2AFC DDM（SST）**：SST（停止机制更符合 stop-signal race/抑制控制模型；可在此任务上开展 SSRT/race 类建模）。
