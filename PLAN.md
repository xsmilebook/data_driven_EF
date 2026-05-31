# Plan

## 修复 THU app 行为指标口径

目标：修复 CPT、EmotionStroop 和 Emotion N-back 指标异常，并将准确率有效试次
与 RT 有效试次分离。原始 `answer` 已人工确认正确，继续作为唯一正确答案来源。

执行项：

- RT 为空的试次仍可进入准确率分母，仅排除 RT 指标；RT 非空但超出 `[0.2, 10]`
  秒范围的试次同时排除准确率和 RT 指标。
- 任务 QC 同时记录 `n_trials_acc_valid` 与 `n_trials_rt_valid`。
- CPT/GNG 的 go/nogo 分类继续使用原始 `answer` 布尔值，保留 CPT 中无 RT 的
  正确 nogo 试次。
- 写入已确认的 EmotionStroop `e1-e32` 条件映射。
- Emotion N-back 先将编号归一化为情绪类别，再比较前 1 或 2 个 trial。
- Emotion2Back 中 item 全空的 sheet 保留整体 ACC/RT，但条件化指标为空并记录
  `nback_item_missing`。
- KT 仅输出 `ACC`。
- 使用 3 个 workbook 做非持久化冒烟测试；随后重跑全量 `clean` 与 `metrics`，
  并复查全空列、常数列和高缺失列。

约束：

- v1 仅实现 THU `app_data`，不扩展 XY、BNU、inventory、demography 或 task-fMRI。
- 不生成或核对 expected answer，不新增 SST/KT 序列审计。
- 不修改用户已有的 `AGENTS.md` 改动。
