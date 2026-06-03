# DDM/HDDM 决策文档

本文件只记录 app 行为任务是否进入 SSM/DDM、高级模型如何编码，以及结果如何汇总。实现细节以代码和后续运行日志为准。

## 总体原则

- 标准 2AFC-RT DDM 只用于有两个响应选项且每个入模试次有有效 RT 的任务。
- 2AFC DDM 默认使用 accuracy coding：`choice=1` 表示正确，`choice=0` 表示错误；此时 `v > 0` 表示证据更快累积到正确反应。
- 多选任务不降维成正确/错误 DDM；保留原始选项，使用 4-choice race 模型。
- Go/NoGo 任务中无 RT 的 NoGo 正确试次不能进入 RT-DDM；若需要 DDM，只解释有按键反应的过程，并与 SDT、commission、omission 并列报告。
- RT 入模口径沿用行为清洗阈值：有效 RT 为 `[0.2, 10]` 秒。

## 任务决策

| Task | 模型决策 | 入模试次 | 主要条件 | 备注 |
| --- | --- | --- | --- | --- |
| FLANKER | 主模型：2AFC 层级 DDM | 全部有效 RT 试次 | `congruency` | 条件均衡，优先解释 `v`；可做 `a ~ congruency` 敏感性分析。 |
| DCCS | 探索：2AFC DDM | 全部有效 RT 试次 | `block` 或整体模型 | 试次数少，mixed 内 switch/repeat 不作为主结论。 |
| SST | 补充：go-only 2AFC DDM | go 且有 RT 试次 | 整体模型 | stop 抑制结论仍来自 SSRT/race，不从 go-only DDM 推断。 |
| CPT | 受限：response-DDM | 有按键且有 RT 的 response trials | `time_bin`；可选 `stimulus_type` | 正确 NoGo 与 omission 无 RT，不能入模；必须并列报告 SDT、commission、omission。 |
| N-back 系列 | 探索：2AFC 层级 DDM | 全部有效 RT 试次 | `load + domain + target_type` | 优先跨 N-back 任务池化；单任务 DDM 仅作敏感性分析。 |
| DT | 主模型：重编码后 2AFC DDM | 按轴向可表示的有效 RT 试次 | Model A: `block + rule`; Model B: `trial_type + rule` | 原始 4-choice，需按 horizontal/vertical 重编码；跨轴错误作为 lapse 剔除并报告。 |
| EmotionSwitch | 主模型：重编码后 2AFC DDM | 按维度可表示的有效 RT 试次 | Model A: `block + rule`; Model B: `trial_type + rule` | 原始 4-choice，需按 gender/emotion 重编码；跨维度错误作为 lapse 剔除并报告。 |
| ColorStroop | 主模型：4-choice race | 全部有效 RT 试次 | `congruency` | 报表可归并为 Target / Word / Other。 |
| EmotionStroop | 主模型：4-choice race | 全部有效 RT 试次 | `congruency` | Target / Word / Other 需要补齐 `e{n}` 映射后再严格报告。 |
| GNG | 不做主 DDM | Go/NoGo 指标 | 无 | 以 SDT、commission、omission、RT 为主；若后续需要，可按 CPT 口径做补充 response-DDM。 |

## 关键编码

### 2AFC DDM

- 必需字段：`subject_id`、`task`、`rt`、`answer`、`key`、`choice`、模型条件变量。
- `answer` 和 `key` 先标准化再判断正确性。
- `choice=1`：`answer == key`；`choice=0`：`answer != key`。
- 条件极不均衡时优先使用层级回归，不做单被试模型主结论。

### DT 与 EmotionSwitch

- DT：根据正确键定义 `rule=axis`，`left/right=horizontal`，`up/down=vertical`。
- EmotionSwitch：根据正确键定义 `rule=dimension`，`female/male=gender`，`happy/sad=emotion`。
- Model A（Mixing）：`pure + mixed`，条件为 `block`，可含 `block:rule`。
- Model B（Switch）：仅 mixed block，条件为 `trial_type`，可含 `trial_type:rule`。
- 被试响应落到另一轴或另一维度时，不编码为普通错误；从 DDM 中剔除，并报告比例。

### CPT

- 仅有实际按键且 RT 有效的试次进入 response-DDM。
- Go 正确按键编码为 `choice=1`；NoGo commission 编码为 `choice=0`。
- `time_bin` 按正式试次顺序分 3 或 4 个等频区间。
- 若 commission 太少，模型退化为 `v ~ time_bin` 或整体模型，不解释 `stimulus_type` 效应。

### N-back

- `load`：oneback=`1`，twoback=`2`。
- `domain`：从任务名提取，如 color、emotion、shape、spatial。
- `target_type`：match vs nonmatch；若 item 信息不足，只进入整体 DDM，不估计 `target_type` 效应。
- match 试次少，`target_type` 效应只作为探索指标；同时报告 hit rate、false alarm rate、d'、criterion、ACC、RT。

### 4-choice race

- ColorStroop choices：`red/green/blue/yellow`。
- EmotionStroop choices：`an/ha/ne/sa`。
- 使用 HSSM/SSMs `race_no_z_4` 作为 4-choice race；这是 LBA4 的替代模型，不把参数直接解释为经典 LBA 的 `A/b`。
- `v0..v3` 对应固定物理选项，不直接等同于 Target/Word/Other；Target/Word/Other 指标从 posterior predictive choice probability 汇总。

## 模型公式

| Task | Model | `v` predictors | `a` predictors | `t0/t` predictors | `z` |
| --- | --- | --- | --- | --- | --- |
| FLANKER | 2AFC DDM | `congruency` | `1`；敏感性：`congruency` | `1` | `1` |
| DCCS | 2AFC DDM（探索） | `block` 或 `1` | `1` | `1` | `1` |
| SST | go-only DDM | `1` | `1` | `1` | `1` |
| CPT | response-DDM | `time_bin`；可选 `stimulus_type` | `1` | `1` | `1` |
| N-back pooled | 2AFC DDM | `load + domain + target_type` | `1` | `1` | `1` |
| N-back task | 敏感性 DDM | `target_type` 或 `1` | `1` | `1` | `1` |
| DT Mixing | 2AFC DDM | `block + rule + block:rule` | 同 `v` | 同 `v` | `1` |
| DT Switch | 2AFC DDM | `trial_type + rule + trial_type:rule` | 同 `v` | 同 `v` | `1` |
| EmotionSwitch Mixing | 2AFC DDM | `block + rule + block:rule` | 同 `v` | 同 `v` | `1` |
| EmotionSwitch Switch | 2AFC DDM | `trial_type + rule + trial_type:rule` | 同 `v` | 同 `v` | `1` |
| ColorStroop | race4 | `congruency` for `v0..v3` | `1` | `1` | 无 |
| EmotionStroop | race4 | `congruency` for `v0..v3` | `1` | `1` | 无 |

## 指标汇总

每个模型至少输出：

- 参数摘要：`mean`、`median`、`sd`、95% HDI、`P(parameter > 0)`。
- 诊断：`r_hat`、ESS、divergence、posterior predictive check。
- 入模 QC：总试次数、入模试次数、排除原因、各条件有效试次数。

主要效应：

- FLANKER：`v_incongruent - v_congruent`。
- CPT：`v_late - v_early`；若可估计，再报告 `v_nontarget - v_target`。
- N-back：`v_2back - v_1back`、`v_match - v_nonmatch`、domain 对比。
- DT/EmotionSwitch：Mixing 为 `mixed - pure`；Switch 为 `switch - repeat`，且 Switch 仅在 mixed block 内解释。
- ColorStroop/EmotionStroop：报告 physical choice drift 的 congruency 效应；Target/Word/Other 从 posterior predictive 汇总。

## 运行建议

- 先用小样本 pilot 确认编码、参数方向和 posterior predictive 合理。
- 全样本分层 SSM 建议在计算节点运行；正式结果使用多链长采样，而不是 pilot 设置。
- 诊断失败的模型不解释参数方向；条件试次数过少时只保留探索性描述。
