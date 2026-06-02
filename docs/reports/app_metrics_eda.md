# THU app_data 指标探索报告

## 分析范围

本报告基于 2026-06-01 重新生成的 THU `app_data` 正式处理产物，覆盖 753 名被试、
18 类任务、1,104,133 行 trial、13,093 行任务 QC 和 124 个被试级宽表指标。分析定位为
描述性 EDA，不执行显著性检验，不修改正式指标口径，也不自动排除异常值候选。

可视化 HTML 位于
`outputs/results/app_data_eda/app_metrics_eda_report.html`。完整聚合表、相关矩阵和
内部定位表位于 `outputs/tables/app_data_eda/`。

## Trial 排除与任务 QC

- 1,027,954 行 trial 具有有效 RT，占全部 trial 的 93.10%。
- `rt_missing` 共 59,329 行，占 5.37%；该现象集中在 CPT，符合 nogo trial 可缺失
  RT 的任务结构。
- `rt_below_min` 共 16,092 行，占 1.46%；`rt_above_max` 共 758 行，占 0.07%；
  未发现 `rt_non_numeric`。
- 任务 QC 共 759 行未通过。失败率较高的任务为 `Emotion2Back`（23.16%）、
  `Spatial2Back`（16.53%）、`GNG`（15.06%）、`Emotion1Back`（14.21%）和
  `Number2Back`（9.30%）。
- raw app 更新后，CPT 中原有的 242 条固定超限值 `1499.8` 已消失。当前
  `rt_above_max` 仅来自 KT（215 条）和 ZYST（543 条）。
- 668 名被试具有全部 18 类任务记录。覆盖率最低的任务为 `Emotion2Back`
  （708/753，94.02%）、`Number2Back`（710/753，94.29%）和 `ZYST`
  （713/753，94.69%）。

### CPT、GNG 与 SST 的 RT 口径说明

EDA 中的 `rt_excluded_rate` 是通用字段检查指标，仅表示 `相对时间(秒)` 是否缺失或
超出 `[0.2, 10]` 秒。它不等价于“最终进入任务 RT 指标计算的 trial 比例”。对于包含
抑制或无反应条件的任务，必须结合任务规则解释。

- CPT 的 `rt_excluded_rate` 为 68.69%，主要来自预期的 NoGo 无反应 trial。CPT 共
  87,840 条 trial，其中 61,488 条为 NoGo；NoGo 中 57,356 条 RT 缺失。RT 缺失的
  NoGo trial 仍可进入 ACC、`NoGo_ACC` 与 `dprime`，因此 CPT 真正因 RT 异常排除出
  准确率分母的比例仅为 1.14%。
- GNG 的 `rt_excluded_rate` 为 1.01%。原始表为 Go 与 NoGo trial 均写入 RT：
  14,740 条 NoGo 中没有 RT 缺失，正确 NoGo 常见 `key=False` 且 RT 约为 0.6004 秒，
  更接近等待窗口结束值。正式 `GNG_Go_RT_Mean` 和 `GNG_Go_RT_SD` 仅使用正确 Go
  trial，因此 NoGo 的窗口结束值不进入 RT 指标。
- SST 的 `rt_excluded_rate` 为 0.31%。原始表为 17,688 条 Stop trial 均写入 RT，
  其中 12,272 条 Stop 成功 trial 未按键，RT 通常约为 2.0014 秒，表示等待窗口结束。
  5,416 条 Stop 失败 trial 中存在两种导出模式：114 条保留了疑似真实按键 RT，
  中位数为 0.826 秒；其余 5,302 条虽然记录了按键，但 RT 仍约为 2.0014 秒，无法从
  当前字段恢复真实按键时刻。正式 `SST_Go_RT_Mean`、`SST_Go_RT_SD` 与 `SST_SSRT`
  仅使用正确 Go RT，因此 Stop trial 的等待窗口值不直接进入 RT 指标。

因此，任务间不应直接比较通用 `rt_excluded_rate`。后续若扩展 EDA，建议增加按任务
规则定义的 `metric_rt_eligible_rate`，并单独报告 SST Stop 失败 trial 中疑似固定等待
窗口值的比例。

## 指标缺失与质量检查

- 340 名被试在 124 个宽表指标上全部非缺失；574 名被试至少具有 90% 的指标。
- 缺失率最高的指标为 `Emotion2Back_Hit_Rate`、`Emotion2Back_FA_Rate` 和
  `Emotion2Back_dprime`，均为 32.40%。`Emotion2Back_ACC`、`RT_Mean` 与
  `RT_SD` 的缺失率均为 27.76%。两者差值来自部分 sheet 的 item 缺失：
  整体 ACC/RT 可保留，但 target 分类指标为空。
- 未发现无穷值、常数指标列或超出 `[0, 1]` 的比例指标。
- Tukey IQR 规则标记出 2,459 行内部异常值候选。该结果用于定位人工复核对象，
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
