# visit1 items: group templates vs app_sequence

本报告对比 `*_item_group1_templates.json` 与 `*_from_app_sequence.json` 的 **items** 是否一致。
注意：仅对比 items；answers 后续应通过推断与人工复核确定，不在本报告中作为一致性标准。

- exact_match tasks: 15/18
- needs_attention tasks: 3/18

## Task summary

| task | template_nonnull_len | app_sequence_len | exact_match | match_ratio(on overlap) | note |
| --- | --- | --- | --- | --- | --- |
| CPT | 120 | 120 | YES | 1.000 |  |
| ColorStroop | 96 | 96 | YES | 1.000 |  |
| DCCS | 40 | 40 | YES | 1.000 |  |
| DT | 128 | 128 | YES | 1.000 |  |
| Emotion1Back | 60 | 60 | YES | 1.000 |  |
| Emotion2Back | 60 | 60 | YES | 1.000 |  |
| EmotionStroop | 96 | 96 | YES | 1.000 |  |
| EmotionSwitch | 128 | 128 | YES | 1.000 |  |
| FLANKER | 96 | 96 | YES | 1.000 |  |
| FZSS | 120 | 120 | YES | 1.000 |  |
| GNG | 100 | 100 | YES | 1.000 |  |
| KT | 1 | 0 | NO |  | app_sequence_items_empty_or_unparsed |
| Number1Back | 60 | 60 | YES | 1.000 |  |
| Number2Back | 60 | 60 | YES | 1.000 |  |
| SST | 0 | 0 |  |  | app_sequence_items_empty_or_unparsed |
| Spatial1Back | 60 | 60 | YES | 1.000 |  |
| Spatial2Back | 60 | 60 | YES | 1.000 |  |
| ZYST | 63 | 0 | NO |  | app_sequence_items_empty_or_unparsed |

## Details (first mismatches)

### KT

- template_len_all=9, template_len_nonnull=1
- app_sequence_len=0
- note: app_sequence_items_empty_or_unparsed

### SST

- template_len_all=96, template_len_nonnull=0
- app_sequence_len=0
- note: app_sequence_items_empty_or_unparsed

### ZYST

- template_len_all=128, template_len_nonnull=63
- app_sequence_len=0
- note: app_sequence_items_empty_or_unparsed

