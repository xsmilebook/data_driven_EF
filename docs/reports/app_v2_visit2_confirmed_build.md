# APP v2：visit2_confirmed 生成与核验

## 目标

生成 `visit2` 的确认版 item+answer 序列文件，放入：

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit2_confirmed.json`

并满足以下口径：

- 使用 `visit2-1125` 的可用任务；
- 对 `visit2` 不包含的任务，使用已确认 `visit3_confirmed` 回填；
- 使用任务特异字段映射，避免通用字段误判。

## 执行命令

```bash
python -m scripts.build_visit2_confirmed_v2 --dataset EFNY_THU
```

脚本：

- `scripts/build_visit2_confirmed_v2.py`

## 关键映射规则

- `CPT`：答案取 `needClickButton`（布尔）
- `DCCS`：答案取 `buttonName`，并做 `0/1 -> Left/Right`
- `KT`：答案从每个 block 的 `answers[].answerPicName` 展开
- `SST`：仅保留前 96 trial（item 置空）
- `ZYST`：使用 `visit2-1125/ZYST_Formal.txt`（不是 `ZYST_Formal2.json`）
- `visit2` 缺失任务：沿用 `visit3_confirmed`（当前对应 `Emotion1Back/Emotion2Back/EmotionStroop/EmotionSwitch/GNG`）

## 输出文件

- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit2_confirmed.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/visit2_confirmed_build_meta.json`
- `data/processed/behavior_data/cibr_app_data_corrected_excel/run_corrected_v2/conformed/manifest.json`（新增 `visit2_confirmed.json` 条目）

## 核验结果（group007）

对 `answer group007`（8 名被试）进行核验，使用与生成一致的比较口径：

- `visit2` 的 13 个任务均为 `8/8` 完全一致：
- `CPT, ColorStroop, DCCS, DT, FLANKER, FZSS, KT, Number1Back, Number2Back, SST(前96), Spatial1Back, Spatial2Back, ZYST`

结论：`visit2_confirmed.json` 与 `group007` 在 visit2 任务上达到全体精确匹配。
