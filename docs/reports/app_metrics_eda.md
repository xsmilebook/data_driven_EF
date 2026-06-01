# THU app_data 指标探索报告

## 分析范围

本报告基于 2026-05-31 生成的 THU `app_data` 正式处理产物，覆盖 753 名被试、18 类
任务、1,104,613 行 trial、13,097 行任务 QC 和 124 个被试级宽表指标。分析定位为
描述性 EDA，不执行显著性检验，不修改正式指标口径，也不自动排除异常值候选。

可视化 HTML 位于
`outputs/results/app_data_eda/app_metrics_eda_report.html`。完整聚合表、相关矩阵和
内部定位表位于 `outputs/tables/app_data_eda/`。

## Trial 排除与任务 QC

- 1,028,100 行 trial 具有有效 RT，占全部 trial 的 93.07%。
- `rt_missing` 共 59,396 行，占 5.38%；该现象集中在 CPT，符合 nogo trial 可缺失
  RT 的任务结构。
- `rt_below_min` 共 16,117 行，占 1.46%；`rt_above_max` 共 1,000 行，占 0.09%；
  未发现 `rt_non_numeric`。
- 任务 QC 共 761 行未通过。失败率较高的任务为 `Emotion2Back`（23.16%）、
  `Spatial2Back`（16.53%）、`GNG`（15.06%）、`Emotion1Back`（14.21%）和
  `Number2Back`（9.30%）。
- 668 名被试具有全部 18 类任务记录。覆盖率最低的任务为 `Emotion2Back`
  （708/753，94.02%）、`Number2Back`（710/753，94.29%）和 `ZYST`
  （713/753，94.69%）。

## 指标缺失与质量检查

- 340 名被试在 124 个宽表指标上全部非缺失；574 名被试至少具有 90% 的指标。
- 缺失率最高的指标为 `Emotion2Back_Hit_Rate`、`Emotion2Back_FA_Rate` 和
  `Emotion2Back_dprime`，均为 32.40%。`Emotion2Back_ACC`、`RT_Mean` 与
  `RT_SD` 的缺失率均为 27.76%。两者差值来自部分 sheet 的 item 缺失：
  整体 ACC/RT 可保留，但 target 分类指标为空。
- 未发现无穷值、常数指标列或超出 `[0, 1]` 的比例指标。
- Tukey IQR 规则标记出 2,464 行内部异常值候选。该结果用于定位人工复核对象，
  不应直接解释为无效数据。

## 指标相关性

Pearson 与 Spearman 矩阵均按成对有效样本计算，并同时导出 pairwise N。任意两指标
的共同有效样本数范围为 448 至 737，中位数为 662。该口径避免将分析限制在仅 340
名全指标完整案例。

Spearman 热图显示，跨任务较强相关主要集中在 RT 指标。例如
`ColorStroop_RT_Mean` 与 `FLANKER_RT_Mean` 的相关为 0.777，
`ColorStroop_RT_Mean` 与 `EmotionStroop_RT_Mean` 的相关为 0.769。整体 ACC 中，
较高相关包括 `DT_ACC` 与 `EmotionSwitch_ACC`（0.684）、`DT_ACC` 与
`FZSS_ACC`（0.679），以及 `Emotion1Back_ACC` 与 `Emotion2Back_ACC`（0.604）。

这些数值用于后续建模前探索，不构成推断性结论。后续若进行假设检验，应单独定义
研究问题、多重比较校正和验证流程。
