# DDM/HDDM 决策文档（PyMC-based）

## 输入与范围
- 输入目录：`D:/projects/data_driven_EF/data/raw/behavior_data/cibr_app_data`
- 文件模式：`*.xlsx`
- 本次运行文件数：10

## 2AFC 合规性检查（全样本）

本节基于 `data/raw/behavior_data/cibr_app_data/` 下全部 `588` 个 Excel 工作簿，汇总各任务 `key(正式阶段被试按键)` 的取值集合（跨被试取并集），用于判断任务是否满足标准 2AFC-RT DDM 的前提。

结论（按 `key` 的 choices 数量）：

| Task | key choices | 2AFC? | choices（key 取值集合） | 备注 |
| --- | --- | --- | --- | --- |
| FLANKER | 2 | 是 | `left/right` | 可作为标准 2AFC-RT DDM 候选 |
| DCCS | 2 | 是 | `left/right` | 2AFC 但试次数较少（尤其 mixed 内） |
| SST | 2 | 部分 | `left/right` | stop 常见无按键；标准 DDM 不适用，需做 go-only DDM（补充） |
| GNG | 2 | 理论上否 | `false/true` | 数据层面有 RT，但任务本质为 go/no-go（建议 SDT；DDM 仅作补充需谨慎解释） |
| CPT | 2 | 部分（受限） | `false/true` | 需纳入 DDM；NoGo 无 RT 不能进入标准 RT-DDM，采用 RT-bearing response DDM，并与 SDT/commission/omission 并列解释 |
| oneback_* / twoback_* | 2 | 是（探索） | `left/right` | 需纳入 2AFC 层级 DDM；match 试次偏少，优先跨 N-back 任务池化建模并谨慎解释条件效应 |
| ColorStroop | 4 | 否 | `red/green/blue/yellow` | **非 2AFC**，应使用多选模型（LBA/race） |
| EmotionStroop | 4 | 否 | `an/ha/ne/sa` | **非 2AFC**，应使用多选模型（LBA/race） |
| DT | 4 | 否（原始） | `left/right/up/down` | **原始为 4-choice**；可按轴向重编码为 2-choice（见下） |
| EmotionSwitch | 4 | 否（原始） | `female/male/happy/sad` | **原始为 4-choice**；可按维度重编码为 2-choice（见下） |

因此：本报告中先前对 `ColorStroop/EmotionStroop/DT/EmotionSwitch` 的 DDM 试算结果应视为“管线通路验证”（legacy），后续应以 LBA 或 2-choice 重编码后的模型结果替代。

## 决策矩阵（依据任务表）
| Task | 纳入试次 | 层级HDDM | 模型（主推） | 备注/风险点 |
| --- | --- | --- | --- | --- |
| FLANKER | 全部 (96; 48C/48I) | 是 | HDDM/层级DDM：v ~ congruency（必要时 a ~ congruency） | 条件均衡，最稳定；也可做单被试分层拟合（不推荐作为主结论）。 |
| ColorStroop | 全部 (96; 24C/72I) | 是（多选） | **4-choice 多选 SSM（HSSM: `race_no_z_4`；LBA4 替代）**（4 accumulators；回归：参数随 congruency 变化） | 报告层面将 choice 归并为：Target（正确选项）、Word（文字诱导选项；仅 incongruent 有意义）、Other（其余两项错误） |
| EmotionStroop | 全部 (96; 24C/72I) | 是（多选） | **4-choice 多选 SSM（HSSM: `race_no_z_4`；LBA4 替代）**（4 accumulators；回归：参数随 congruency 变化） | 需要在试次级数据中解析“Target vs Word”；当前 `item=e{n}` 仅支持 congruency 判定，需补充映射表后才能稳定构造 Word 诱导选项 |
| DT | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（重编码后） | **2-choice 层级 DDM（按轴向重编码）**：建立两个对照模型（A/B），均纳入 `rule=axis`，并允许 `v/a/t0` 随条件变化（含交互） | Model A：Mixing（pure vs mixed）；Model B：Switch（repeat vs switch，仅 mixed）。跨轴错误按 lapse/outlier 处理并报告比例。 |
| EmotionSwitch | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（重编码后） | **2-choice 层级 DDM（按维度重编码）**：建立两个对照模型（A/B），均纳入 `rule=dimension`，并允许 `v/a/t0` 随条件变化（含交互） | Model A：Mixing（pure vs mixed）；Model B：Switch（repeat vs switch，仅 mixed）。跨维度错误按 lapse/outlier 处理并报告比例。 |
| DCCS | pure 20 + mixed 20 = 40；mixed 内 10R/10S | 是（探索） | 整体 DDM 或 v ~ block_type | mixed 内每格 10 太少；switch-DDM 基本不可识别。 |
| SST | 主模型用 go+stop（race/SSRT）；DDM 仅用 go RT | ①否（标准 DDM）②是（Go-only DDM） | 主推：Stop-signal race/SSRT；补充：**Go-only 2AFC DDM（仅 go trials）** | stop 无反应不满足 DDM；go-only DDM 用于刻画 go 决策（需与 SSRT/race 分析并列呈现） |
| CPT | Go/NoGo；DDM 仅纳入有 RT 的 response trials | 是（受限） | **CPT response-DDM（RT-bearing trials）**：`v ~ stimulus_type + time_bin`（若 commission 过少则退化为 `v ~ time_bin` 或整体模型） | DDM 解释有按键反应的证据积累/谨慎性/非决策时间；无按键 NoGo、omission 不能进入 RT-DDM，仍需并列报告 SDT、commission、omission。 |
| GNG | Go/NoGo（NoGo 无 RT） | 否（主模型） | SDT(d'+c) + commission/omission（可加 time-on-task）；DDM 仅在用户要求时作为与 CPT 同口径的补充 | 单键 Go/NoGo 不符合标准 2AFC-RT DDM；NoGo 无 RT。 |
| N-back 系列 | 60（match 较少） | 是（探索） | **2AFC 层级 DDM**：优先跨 N-back 任务池化，`v ~ load + domain + target_type`；单任务敏感性模型可用 `v ~ target_type` 或整体模型 | 使用 `left/right` 物理 2-choice 与 accuracy coding；match 稀少，条件效应只作为探索结果，并与 SDT(d'+c)、RT/ACC、负荷/域比较并列呈现。 |

## 关键重编码与报表归并规则（保证 2AFC / 多选一致性）

本节将“原始 choices（任务真实按键集合）”与“用于建模的数据结构”显式对齐，避免在 2AFC-RT DDM 上违反基本假设。

### 统一试次级入模规则

所有 SSM/DDM 拟合均以清洗后的正式试次为输入，不直接使用练习试次或无法确认任务条件的试次。入模前需生成一张 trial-level 表，至少包含：

- `subject_id`：被试 ID。
- `task`：任务名。
- `trial_index`：被试内正式试次顺序。
- `rt`：反应时，单位为秒。
- `answer`：被试实际响应。
- `key`：该试次正确响应。
- `accuracy`：`answer == key`。
- `condition`：任务主条件，例如 `congruency`、`block`、`trial_type`。
- `rule`：仅用于 DT/EmotionSwitch，分别为 `axis` 或 `dimension`。
- `choice_model`：模型实际使用的 choice 编码。
- `model_include`：该试次是否进入对应模型。
- `exclude_reason`：未入模原因；常见值包括 `practice_trial`、`missing_rt`、`rt_out_of_range`、`missing_choice`、`invalid_choice_for_model`、`cross_rule_lapse`、`condition_missing`。

反应时规则：

- SSM/DDM 拟合要求 `rt` 非空、有限且大于 0；没有 RT 的试次不能进入 RT-based SSM。
- 若上游行为清洗已定义统一 RT 合理范围，则沿用同一范围；当前行为指标口径为 `[0.2, 10]` 秒。低于下限或高于上限的试次从 SSM 拟合中剔除，并在质量表中报告数量与比例。
- RT 单位必须在入模前统一为秒；若原始表为毫秒，转换后再应用范围阈值。

choice 规则：

- `answer` 与 `key` 均先做标准化（去空格、大小写统一、布尔值统一为小写字符串或布尔型），再比较正确性。
- 2AFC DDM 主模型采用 accuracy coding：`choice_model=1` 表示正确边界，`choice_model=0` 表示错误边界；此时 `v > 0` 表示证据更快累积到正确反应。若研究问题涉及左右手/物理按键偏置，再另做 stimulus coding 敏感性分析。
- 多选 race4 模型保留物理响应选项作为 `choice_model`，不得先归并为正确/错误后拟合 2AFC DDM。
- 对 DT/EmotionSwitch，跨轴或跨维度错误不是对应 2-choice 子任务中的可选错误边界，不能编码为普通错误；这类试次只作为 `cross_rule_lapse` 质量指标统计，并从 2AFC DDM 中剔除。

被试纳入规则：

- 标准 2AFC task 中，每个 task×model×subject 至少需要同时存在两个 choice 结局，且关键条件至少各有可用试次；否则该被试不用于该模型的个体层信息估计。
- 对 CPT/N-back 这类错误或 match 试次稀少的任务，可在层级模型中保留仅有单一 choice 结局的被试以贡献 RT 分布信息，但不解释该被试的个体层 choice/条件效应；报告中必须列出每个被试和每个条件的正确/错误或 match/nonmatch 计数。
- 若单个条件有效试次数过低（建议阈值：每格 `< 5`），该条件的个体层效应不单独解释；仍可在层级模型中贡献组水平信息，但需在报告中列出每格试次数分布。
- 每个模型输出一张 trial QC 汇总：`n_total`、`n_model_include`、`n_excluded`、各 `exclude_reason` 计数、各条件有效试次数、跨轴/跨维度错误比例、RT 中位数与四分位数。

### DT：`left/right/up/down` → 两个 2-choice 子任务（轴向重编码）
- 轴向判定（以正确按键为准）：若 `key ∈ {left,right}` 记为 `axis=horizontal`；若 `key ∈ {up,down}` 记为 `axis=vertical`。
- `rule` 定义：本任务中 `rule := axis`（horizontal vs vertical），用于控制不同规则/轴向的基线差异，并允许与实验条件交互。
- 2-choice 化：在每个 `axis` 内，将 `choice` 限定为该轴的两键（horizontal: left vs right；vertical: down vs up）。
- 响应处理（关键）：若被试响应 `answer` 落在**另一条轴**（例如 horizontal 试次答了 up），该响应在 2-choice DDM 中不可表示，建议作为 `lapse/outlier` 计数并从 DDM 拟合数据中剔除；同时在报告中单独汇报其比例（质量指标）。
- 条件变量：
  - `block ∈ {pure,mixed}`：用于 Mixing 分析（Model A）。
  - `trial_type ∈ {repeat,switch}`：用于 Switch 分析（Model B；仅 mixed 内有定义）。
- 建模策略建议：优先在一个层级 DDM 中加入 `axis` 哑变量做控制（axis 作为协变量，或在后续做分轴敏感性分析），避免把两种语义不同的 2-choice 直接无区分混合。

### EmotionSwitch：`female/male/happy/sad` → 两个 2-choice 子任务（维度重编码）
- 维度判定（以正确按键为准）：若 `key ∈ {female,male}` 记为 `dimension=gender`；若 `key ∈ {happy,sad}` 记为 `dimension=emotion`。
- `rule` 定义：本任务中 `rule := dimension`（gender vs emotion），用于控制不同规则/维度的基线差异，并允许与实验条件交互。
- 2-choice 化：在每个 `dimension` 内仅保留对应两键（gender: female vs male；emotion: happy vs sad）。
- 响应处理：若被试响应跨维度（例如 emotion 试次答了 female），同 DT 处理为 `lapse/outlier` 计数并从 DDM 拟合数据中剔除，同时报告比例。
- 条件变量：
  - `block ∈ {pure,mixed}`：用于 Mixing 分析（Model A）。
  - `trial_type ∈ {repeat,switch}`：用于 Switch 分析（Model B；仅 mixed 内有定义）。
- 建模策略建议：建议将 `rule` 纳入模型并允许与 block/trial_type 交互；同时建议报告两维度分别的基础行为差异（如 ACC/RT 分布）以辅助解释。

### SST：go-only 2AFC DDM（补充模型）
- 仅纳入 go 试次（需能稳定判定 go/stop；并排除 stop 无按键/无 RT 试次），在 go 试次内按 `left/right` 做 2AFC DDM。
- go-only DDM 的作用定位为“go 决策过程刻画”，必须与 SSRT/race 模型的 stop 抑制结论并列解释。

### CPT：RT-bearing response DDM（受限模型）
- 任务类型为 Go/NoGo；若 `key=true` 记为 `stimulus_type=target/go`，若 `key=false` 记为 `stimulus_type=nontarget/nogo`。
- 标准 RT-DDM 只能使用有实际按键和有效 RT 的试次：Go 正确按键与 NoGo commission error 可入模；正确 NoGo 无 RT 与 Go omission 无 RT 不能入模。
- `choice_model` 采用 accuracy coding：Go 正确按键为 `1`，NoGo commission 为 `0`。若存在其他有 RT 的异常响应，需先按上游清洗规则决定是否为有效 response trial。
- `time_bin` 建议按正式试次顺序分为 3 或 4 个等频区间，用于刻画 CPT 的 time-on-task/vigilance 变化；若每格 commission 数不足，则仅保留 `time_bin` 或整体截距模型，不估计 `stimulus_type` 条件效应。
- 报告定位：CPT DDM 是 response-conditioned 模型，不能解释“成功不按键”的潜伏过程；主报告必须同时给出 SDT(d'+c)、commission rate、omission rate 和 RT 描述指标。

### N-back 系列：2AFC 层级 DDM（探索模型）
- 原始 `left/right` 为标准 2-choice 响应；统一采用 accuracy coding，`choice_model=1` 表示响应等于 `key`，`choice_model=0` 表示响应不等于 `key`。
- `target_type`：若当前试次与前 `n` 个试次匹配，记为 `match`；否则为 `nonmatch`。若上游无法稳定判定 match/nonmatch（例如 item 全空），可进入整体 DDM，但该试次不参与 `target_type` 效应估计。
- `load`：oneback 记为 `1`，twoback 记为 `2`。
- `domain`：从任务名提取，例如 color、emotion、shape、spatial；命名需与行为指标表保持一致。
- 主模型建议跨 N-back 系列池化，以 `load`、`domain`、`target_type` 作为协变量；单任务模型仅作为敏感性分析或任务级描述。
- 由于 match 试次数较少，`target_type` 相关 DDM 参数只作为探索性指标；正式报告中仍需并列给出 hit rate、false alarm rate、d'、criterion、ACC 与 RT。

### ColorStroop / EmotionStroop：4-choice 多选 SSM（race4/LBA 替代）+ 报表三类归并
- 拟合层面：保留**原始 4-choice** 作为四个 accumulators（ColorStroop: `red/green/blue/yellow`；EmotionStroop: `an/ha/ne/sa`），以 congruency 为主要条件变量（当前实现使用 HSSM: `race_no_z_4`）。
- 报表层面：将每个试次的响应按以下三类归并，用于更直观的错误结构总结：
  - `Target`：响应等于正确选项（`answer == key`）。
  - `Word`：仅对 incongruent 试次定义；响应等于“文字诱导选项”（ColorStroop 可从 `item` 解析 Text 颜色；EmotionStroop 需从试次信息解析 word 对应类别）。
  - `Other`：其余两项错误选项。
- 可实现性提醒：ColorStroop 的 `item` 形如 `Pic_<Color>_Text_<Color>`，可直接构造 Word；EmotionStroop 当前 `item=e{n}` 仅支持 congruency 判定，需补充“每个 e{n} 对应的 target 与 word 类别”映射表（或从原始表格中读取显式字段）后才能严格区分 `Word` 与 `Other`。

## 模型设计总览（按任务×参数）

本节给出“每个任务每个参数允许随哪些条件变化”的统一规范，用于后续实现对照模型比较与全样本 SLURM 计算。记号约定：

- 2AFC DDM 参数：`v`（drift）、`a`（boundary）、`t0`（non-decision time）、`z`（starting point）。
- 多选任务使用 4-choice race（`race_no_z_4`；作为 LBA4 的可行替代），以 `v0..v3`（各 accumulator drift）、`a`、`t0=t` 为主。
- 层级结构：默认对“参与比较或解释的关键参数”加入被试随机截距（random effects）。对 DDM 为 `v/a/t0/z`（至少 `v`），对 4-choice race 为 `v0..v3`（至少）与 `t`（可选）。

### 2AFC DDM（层级回归）设计

| Task | Model | 试次范围 | predictors（v） | predictors（a） | predictors（t0） | predictors（z） |
| --- | --- | --- | --- | --- | --- | --- |
| FLANKER | 基线/主模型 | 全部 | `congruency` | `1`（可选敏感性：`congruency`） | `1` | `1` |
| SST | go-only DDM | go trials | `1`（如需可扩展：`hand`/`congruency` 等） | `1` | `1` | `1` |
| CPT | response-DDM（受限） | RT-bearing response trials | `stimulus_type + time_bin`（commission 过少时：`time_bin` 或 `1`） | `1` | `1` | `1` |
| N-back 系列 | pooled DDM（主探索） | 全部 RT-valid trials | `load + domain + target_type` | `1` | `1` | `1` |
| N-back 系列 | per-task sensitivity | 单个 N-back task | `target_type`（match 过少时：`1`） | `1` | `1` | `1` |
| DT | Model A：Mixing | pure+mixed | `block + rule + block:rule` | `block + rule + block:rule` | `block + rule + block:rule` | `1` |
| DT | Model B：Switch | mixed only | `trial_type + rule + trial_type:rule` | `trial_type + rule + trial_type:rule` | `trial_type + rule + trial_type:rule` | `1` |
| EmotionSwitch | Model A：Mixing | pure+mixed | `block + rule + block:rule` | `block + rule + block:rule` | `block + rule + block:rule` | `1` |
| EmotionSwitch | Model B：Switch | mixed only | `trial_type + rule + trial_type:rule` | `trial_type + rule + trial_type:rule` | `trial_type + rule + trial_type:rule` | `1` |
| DCCS | 仅探索（不建议） | 全部 | `block`（或 `1`） | `1` | `1` | `1` |

实现要点：
- `DT` 的 `rule` 为 `axis`（horizontal/vertical）；`EmotionSwitch` 的 `rule` 为 `dimension`（gender/emotion）。
- Model B（Switch）默认仅在 mixed block 内拟合（repeat/switch 有定义）；与 Model A（Mixing）是并行对照，不做“同一模型强行同时放入 block 与 trial_type”的主结论。

### 4-choice 多选 SSM（层级回归；LBA4 的可行替代）

本项目当前使用 HSSM/SSMs 内置的 `race_no_z_4`（4-choice race，无 starting-point bias 参数）作为 4-choice LBA 的可行替代：二者同属“多 accumulator 的赛跑式序贯采样模型”，但**参数含义与先验设定并不等价**；因此在报告中需明确写为 “race4（LBA 替代）”，并避免把参数名直接解释为经典 LBA 的 `A/b`。

| Task | Choices | predictors（v0..v3；各 accumulator） | predictors（a） | predictors（t0=t） | 报表归并 |
| --- | --- | --- | --- | --- | --- |
| ColorStroop | 4（颜色键） | `congruency` | `1` | `1` | Target / Word / Other |
| EmotionStroop | 4（情绪键） | `congruency` | `1` | `1` | Target / Word / Other（需补齐 target/word 映射） |

说明：
- 这里的 `v0..v3 ~ congruency` 表示：四个 accumulator 的 drift 允许随 congruency 变化；`a` 与 `t` 默认不随条件变化以控制复杂度（如后验预测提示系统性偏差，可再扩展）。
- `v0..v3` 对应固定物理选项，而不是自动对应 Target/Word/Other 角色。因此 race4 参数解释应写为“某物理响应选项的 drift 随 congruency 改变”；Target/Word/Other 的主要报告指标应来自 posterior predictive choice probability，而不是直接把某个 `v{k}` 解释为 Target 或 Word drift。

## 高级模型指标计算口径

本节定义从 posterior、posterior predictive 和 trial QC 中提取的统一指标。除非特别说明，所有汇总均在 task×model 层面输出组水平结果，并保留 subject-level posterior 或 posterior predictive 汇总用于后续个体差异分析。

### 1. 参数后验摘要

每个可解释参数或回归系数均输出：

- `mean`、`median`、`sd`。
- `hdi_2.5%`、`hdi_97.5%`（95% HDI）。
- `p_gt_0 = P(parameter > 0 | data)` 与 `p_lt_0 = P(parameter < 0 | data)`。
- `effect_direction`：若 `p_gt_0 >= 0.975` 记为 `positive`；若 `p_lt_0 >= 0.975` 记为 `negative`；否则记为 `uncertain`。

参数解释规则：

- 2AFC accuracy-coded DDM 中，`v` 越大表示越快趋向正确边界；负向条件效应表示该条件降低证据积累效率。
- `a` 越大表示反应更谨慎，通常对应更慢但可能更准确的反应。
- `t0` 越大表示非决策过程更长，例如知觉编码或运动执行时间更长。
- `z` 若固定为 `1` 或不作为主效应解释，则仅作为模型结构项记录；若估计 `z`，必须说明其编码边界含义。
- race4 中 `a` 与 `t` 的解释与 DDM 类似；`v0..v3` 只解释固定物理选项的 accumulator drift。

### 2. 条件效应与对照指标

条件效应均使用 posterior draw 逐抽样计算，再汇总均值、95% HDI 和方向概率，避免只用点估计相减。

FLANKER：

- 冲突效应：`Δv_incongruent = v_incongruent - v_congruent`。若为负，解释为 incongruent 条件降低证据积累效率。
- 敏感性模型若包含 `a ~ congruency`，同步计算 `Δa_incongruent`；若 `Δa` 为正，解释为冲突条件下更保守的边界设置。

DT / EmotionSwitch：

- Mixing effect（Model A）：`Δparam_mixing(rule) = param_mixed,rule - param_pure,rule`。
- Rule-averaged mixing effect：对两个 rule 的 `Δparam_mixing(rule)` 取 posterior draw 层面的平均。
- Rule interaction：`Δparam_mixing(rule_1) - Δparam_mixing(rule_2)`，用于判断 mixing cost 是否依赖轴向/维度。
- Switch effect（Model B）：`Δparam_switch(rule) = param_switch,rule - param_repeat,rule`，仅在 mixed block 内计算。
- Rule-averaged switch effect 与 rule interaction 的计算同上。
- 主报告优先解释 `v`、`a`、`t0` 的 mixing/switch effect；若某个参数后验方向不确定，不做强结论。

DCCS：

- 仅输出探索性 `Δv_block = v_mixed - v_pure` 或整体参数摘要。
- 不报告 mixed 内 switch/repeat 参数效应作为主结果，因为每格试次数不足以支持稳定识别。

SST：

- go-only DDM 只解释 go 试次的证据积累、谨慎性和非决策时间。
- stop 抑制能力不从 go-only DDM 推断；SSRT 或 stop-signal race 指标需独立计算并并列报告。

CPT：

- response-DDM 的主效应为 `time_bin`：`Δv_time = v_late - v_early`，用于描述持续注意或警觉性下降是否体现在证据积累效率上。
- 若 commission 数量足够，可计算 `Δv_nontarget = v_nontarget - v_target`；该指标只解释有按键反应中的错误倾向，不解释正确 NoGo 的抑制过程。
- `Δa_time` 或 `Δt0_time` 默认不作为主模型；若 posterior predictive 显示 RT 分位数无法复现，再作为敏感性模型扩展。
- DDM 指标必须与 `commission_rate`、`omission_rate`、`dprime`、`criterion` 并列报告；若 commission 太少导致 choice 模型不可识别，仍输出 RT-only/整体参数摘要并标记 `choice_effect_unstable`。

N-back：

- 负荷效应：`Δv_load = v_2back - v_1back`。若为负，解释为 2-back 相比 1-back 降低证据积累效率。
- 目标类型效应：`Δv_match = v_match - v_nonmatch`。由于 match 稀少，仅作为探索指标；需同时报告 posterior HDI 与 match/nonmatch 有效试次数。
- 领域效应：以一个 domain 为参考水平，输出其他 domain 的 `Δv_domain`；若样本量允许，也可报告 `load:domain` 交互。
- 单任务敏感性模型只用于检查 pooled DDM 的方向是否稳定，不作为优先结论来源。
- N-back DDM 与 SDT 指标互补：DDM 描述有 RT 的选择过程，d' 与 criterion 描述 match/nonmatch 检测敏感性和反应偏向。

ColorStroop / EmotionStroop：

- 参数层面报告 `v0..v3`、`a`、`t` 的 congruency 效应，但文字解释限定为固定物理响应选项。
- 行为角色层面从 posterior predictive 计算 `P(Target)`、`P(Word)`、`P(Other)`，并输出 `ΔP(Target) = P(Target|incongruent) - P(Target|congruent)`。
- Word 诱导效应仅在 incongruent 试次定义：`Word intrusion = P(Word|incongruent) - P(Other_each|incongruent)`；其中 `Other_each` 为两个 Other 选项的平均预测概率。
- 若 EmotionStroop 未补齐 target/word 映射，只能报告 `P(correct)` 与四选项物理 choice 概率，不报告 Word intrusion。

### 3. Posterior predictive 指标

每个 task×model 至少生成 posterior predictive summary，用于确认模型是否同时复现 RT 与 choice：

- `pred_acc`：预测正确率或 Target 选择概率。
- `pred_choice_prob`：各 choice 的预测概率。
- `pred_rt_mean`、`pred_rt_median`。
- `pred_rt_q10`、`pred_rt_q50`、`pred_rt_q90`。
- `pred_error_rt_median`：错误试次预测 RT 中位数；若错误试次过少则记为空。
- `observed_minus_predicted`：观测值减预测均值，用于定位系统性偏差。

条件差异的 RT 指标应基于 posterior predictive 分布计算，例如：

- FLANKER RT cost：`RT_incongruent - RT_congruent`。
- Mixing RT cost：`RT_mixed - RT_pure`。
- Switch RT cost：`RT_switch - RT_repeat`。
- CPT time-on-task RT change：`RT_late - RT_early`。
- N-back load RT cost：`RT_2back - RT_1back`；N-back target RT contrast：`RT_match - RT_nonmatch`。

这些 RT cost 是模型预测层面的描述指标，不等同于 `t0` 或 `a` 的单一参数效应；解释时需与 `v/a/t0` 的后验效应一起报告。

### 4. 模型质量与比较

正式报告前需记录以下诊断：

- MCMC 收敛：`r_hat <= 1.01`、bulk/tail ESS 足够；若未达到，标记为 `diagnostic_fail`，不解释参数方向。
- 采样异常：divergence 数、最大 tree depth、E-BFMI；存在明显异常时先调整采样或先验，不直接比较模型。
- Posterior predictive check：按条件比较 choice probability 与 RT 分位数，确认模型未系统性低估错误率或 RT 尾部。
- 模型比较：仅在诊断合格后使用 LOO/WAIC 或 out-of-sample predictive score 比较 null/effect 模型；报告 `elpd_diff` 与标准误，不仅报告“某模型更优”。

模型命名建议：

- `flanker_congruency_ddm`
- `dt_mixing_ddm`、`dt_switch_ddm`
- `emotionswitch_mixing_ddm`、`emotionswitch_switch_ddm`
- `colorstroop_race4_congruency`
- `emotionstroop_race4_congruency`
- `sst_go_only_ddm`
- `cpt_response_ddm`
- `nback_pooled_ddm`
- `nback_task_ddm`

## 本次模型计算设置
- 后端：HSSM（PyMC-based hierarchical SSM） + `nuts_numpyro` 采样
- draws=40, tune=40, chains=1, seed=1
- 层级结构（目标）：对 `v/a/t/z` 均引入被试随机截距（random effects）。除 `DT/EmotionSwitch` 的 Model A/B 明确要求 `a` 与 `t0` 随条件变化外，其余任务默认仅允许 `v` 随条件变化（必要时 FLANKER 可做 `a ~ congruency` 的敏感性分析）。
- 说明：计算脚本会将 posterior traces（netcdf）与轻量 summary（CSV/JSON）写入 `data/processed/table/metrics/ssm/`，便于集群计算后下载到本地做二次分析与汇总。
- 追溯性保存：每个 task×model（null/effect）保存 posterior traces（`InferenceData` netcdf），并另存一份轻量 summary（便于下载后本地汇总与复核）。

## 计算结果摘要（效果评估）

## 复现方式
```bash

```

## 全样本运行建议（规划）
- 先完成小规模 pilot（例如 `--max-files 10`），确认依赖与模型可跑通、参数符号与数量级合理。
- 全样本层级 SSM 计算耗时长，建议在可用计算节点上运行，并逐步提高采样长度（分层 SSM 建议至少 tune≥1000–2000, draws≥1000, chains≥4），并通过 SLURM array 并行不同 task×model：


## 风险与注意事项（规划）
- 层级 SSM 对计算资源敏感：建议先用 `--max-files` 进行 pilot，确认模型可运行后再扩展到全样本。
- 对多选任务：`ColorStroop/EmotionStroop` 应使用 4-choice 多选 SSM（当前实现：`race_no_z_4`；LBA4 替代）；避免将其降维为“正确/错误”的 2AFC DDM 解释。
- 对切换任务：`DT/EmotionSwitch` 需先做 2-choice 重编码（并剔除跨轴/跨维度错误），否则 2AFC DDM 解释不成立。
- 对 CPT：DDM 仅覆盖有按键且有 RT 的 response trials，不能替代 NoGo 成功抑制或 omission 指标；若 commission 极少，`stimulus_type` 效应不可解释。
- 对 N-back：match 试次偏少，优先使用跨任务层级池化；单任务 DDM 只作为敏感性分析，且必须报告每格有效试次数。
- 对条件极不均衡任务：应避免单被试分层拟合，优先层级回归形式，并在 posterior predictive 中检查 choice 概率是否合理。
- 对切换任务（DT/EmotionSwitch）：建议采用两个并行对照模型（Model A：Mixing；Model B：Switch），避免在同一模型中同时放入 `block` 与 `trial_type` 导致解释混淆；并在报告中明确 Model B 仅基于 mixed block。 
