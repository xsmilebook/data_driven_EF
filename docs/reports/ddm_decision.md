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
| CPT | 2 | 否 | `false/true` | NoGo 基本无 RT（不满足 2AFC-RT DDM） |
| oneback_* / twoback_* | 2 | 不建议 | `left/right` | 虽 2-choice，但 match 试次偏少；DDM 解释力弱 |
| ColorStroop | 4 | 否 | `red/green/blue/yellow` | **非 2AFC**，应使用多选模型（LBA/race） |
| EmotionStroop | 4 | 否 | `an/ha/ne/sa` | **非 2AFC**，应使用多选模型（LBA/race） |
| DT | 4 | 否（原始） | `left/right/up/down` | **原始为 4-choice**；可按轴向重编码为 2-choice（见下） |
| EmotionSwitch | 4 | 否（原始） | `female/male/happy/sad` | **原始为 4-choice**；可按维度重编码为 2-choice（见下） |

因此：本报告中先前对 `ColorStroop/EmotionStroop/DT/EmotionSwitch` 的 DDM 试算结果应视为“管线通路验证”（legacy），后续应以 LBA 或 2-choice 重编码后的模型结果替代。

## 决策矩阵（依据任务表）
| Task | 纳入试次 | 层级HDDM | 模型（主推） | 备注/风险点 |
| --- | --- | --- | --- | --- |
| FLANKER | 全部 (96; 48C/48I) | 可选（推荐做统一管线） | HDDM/层级DDM：v ~ congruency（必要时 a ~ congruency） | 条件均衡，最稳定；也可做单被试分层拟合（不推荐作为主结论）。 |
| ColorStroop | 全部 (96; 24C/72I) | 是（多选） | **4-choice 多选 SSM（HSSM: `race_no_z_4`；LBA4 替代）**（4 accumulators；回归：参数随 congruency 变化） | 报告层面将 choice 归并为：Target（正确选项）、Word（文字诱导选项；仅 incongruent 有意义）、Other（其余两项错误） |
| EmotionStroop | 全部 (96; 24C/72I) | 是（多选） | **4-choice 多选 SSM（HSSM: `race_no_z_4`；LBA4 替代）**（4 accumulators；回归：参数随 congruency 变化） | 需要在试次级数据中解析“Target vs Word”；当前 `item=e{n}` 仅支持 congruency 判定，需补充映射表后才能稳定构造 Word 诱导选项 |
| DT | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（重编码后） | **2-choice 层级 DDM（按轴向重编码）**：建立两个对照模型（A/B），均纳入 `rule=axis`，并允许 `v/a/t0` 随条件变化（含交互） | Model A：Mixing（pure vs mixed）；Model B：Switch（repeat vs switch，仅 mixed）。跨轴错误按 lapse/outlier 处理并报告比例。 |
| EmotionSwitch | pure 64 + mixed 64 = 128；switch 仅 mixed | 是（重编码后） | **2-choice 层级 DDM（按维度重编码）**：建立两个对照模型（A/B），均纳入 `rule=dimension`，并允许 `v/a/t0` 随条件变化（含交互） | Model A：Mixing（pure vs mixed）；Model B：Switch（repeat vs switch，仅 mixed）。跨维度错误按 lapse/outlier 处理并报告比例。 |
| DCCS | pure 20 + mixed 20 = 40；mixed 内 10R/10S | 否（或仅探索） | 不建议做 switch-DDM；如一定要做，仅做整体 DDM 或 v ~ block_type | mixed 内每格 10 太少；switch-DDM 基本不可识别。 |
| SST | 主模型用 go+stop（race/SSRT）；DDM 仅用 go RT | ①否（标准 DDM）②是（Go-only DDM） | 主推：Stop-signal race/SSRT；补充：**Go-only 2AFC DDM（仅 go trials）** | stop 无反应不满足 DDM；go-only DDM 用于刻画 go 决策（需与 SSRT/race 分析并列呈现） |
| GNG / CPT | Go/NoGo（NoGo 无 RT） | 否 | 不做标准 DDM；用 SDT(d'+c) + commission/omission（可加 time-on-task） | 单键 Go/NoGo 不符合标准 2AFC-RT DDM；NoGo 无 RT。 |
| N-back 系列 | 60（match 很少） | 否（不建议） | SDT(d'+c) + RT/ACC + 负荷/域比较 | match 太少；分层 DDM 不可行，整体 DDM 解释弱。 |

## 关键重编码与报表归并规则（保证 2AFC / 多选一致性）

本节将“原始 choices（任务真实按键集合）”与“用于建模的数据结构”显式对齐，避免在 2AFC-RT DDM 上违反基本假设。

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

## 本次模型计算设置
- 后端：HSSM（PyMC-based hierarchical SSM） + `nuts_numpyro` 采样
- draws=40, tune=40, chains=1, seed=1
- 层级结构（目标）：对 `v/a/t/z` 均引入被试随机截距（random effects）。除 `DT/EmotionSwitch` 的 Model A/B 明确要求 `a` 与 `t0` 随条件变化外，其余任务默认仅允许 `v` 随条件变化（必要时 FLANKER 可做 `a ~ congruency` 的敏感性分析）。
- 说明：计算脚本会将 posterior traces（netcdf）与轻量 summary（CSV/JSON）写入 `data/processed/table/metrics/ssm/`，便于集群计算后下载到本地做二次分析与汇总。
- 追溯性保存：每个 task×model（null/effect）保存 posterior traces（`InferenceData` netcdf），并另存一份轻量 summary（便于下载后本地汇总与复核）。

## 计算结果摘要（效果评估）
说明：下表为早期 DDM 试算（用于验证管线可运行与效应符号）。其中 `DT/EmotionSwitch` 需先完成 2-choice 重编码（并剔除跨轴/跨维度错误）；`ColorStroop/EmotionStroop` 应以 4-choice 多选 SSM（`race_no_z_4`；LBA 替代）结果替代。

| Task | Model | N_subjects | N_trials | Beta(v_congruency) | Beta(v_block_mixed) | Beta(v_is_switch) | LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FLANKER | HDDM (hier): v ~ congruency | 10 | 960 | -0.257 [-0.501, -0.106], P(>0)=0.00 |  |  | elpd_loo=495.2, p_loo=17.1 |
| ColorStroop | HDDM (hier): v ~ congruency | 10 | 960 | -0.568 [-0.702, -0.421], P(>0)=0.00 |  |  | elpd_loo=-303.2, p_loo=13.8 |
| EmotionStroop | HDDM (hier): v ~ congruency | 10 | 960 | -0.274 [-0.432, -0.136], P(>0)=0.00 |  |  | elpd_loo=-1339.4, p_loo=12.7 |
| DT | HDDM (hier): v ~ block_mixed | 10 | 1280 |  | -0.348 [-0.463, -0.242], P(>0)=0.00 |  | elpd_loo=-1772.3, p_loo=13.8 |
| EmotionSwitch | HDDM (hier): v ~ block_mixed | 10 | 1280 |  | -0.436 [-0.572, -0.309], P(>0)=0.00 |  | elpd_loo=-1582.0, p_loo=13.8 |
| DT | HDDM (hier): v ~ is_switch (mixed only) | 10 | 640 |  |  | -0.179 [-0.329, -0.028], P(>0)=0.03 | elpd_loo=-1016.4, p_loo=16.6 |
| EmotionSwitch | HDDM (hier): v ~ is_switch (mixed only) | 10 | 640 |  |  | -0.320 [-0.473, -0.140], P(>0)=0.00 | elpd_loo=-911.9, p_loo=14.3 |

## 复现方式
```bash
# 2AFC DDM（示例：DT Mixing 模型；job-index 4）
python -m scripts.fit_ssm_task --dataset EFNY --config configs/paths.yaml --job-index 4 --max-files 10 --draws 40 --tune 40 --chains 1 --seed 1

# 4-choice 多选 SSM（示例：ColorStroop congruency 模型；job-index 1）
python -m scripts.fit_race4_task --dataset EFNY --config configs/paths.yaml --job-index 1 --max-files 10 --draws 40 --tune 40 --chains 1 --seed 1
```

## 全样本运行建议（规划）
- 先完成小规模 pilot（例如 `--max-files 10`），确认依赖与模型可跑通、参数符号与数量级合理。
- 全样本层级 SSM 计算耗时长，建议在可用计算节点上运行，并逐步提高采样长度（如 draws≥500, tune≥500, chains≥2），并通过 SLURM array 并行不同 task×model：
```bash
sbatch scripts/submit_hpc_ssm.sh
sbatch scripts/submit_hpc_race4.sh
```

## 风险与注意事项（规划）
- 层级 SSM 对计算资源敏感：建议先用 `--max-files` 进行 pilot，确认模型可运行后再扩展到全样本。
- 对多选任务：`ColorStroop/EmotionStroop` 应使用 4-choice 多选 SSM（当前实现：`race_no_z_4`；LBA4 替代）；避免将其降维为“正确/错误”的 2AFC DDM 解释。
- 对切换任务：`DT/EmotionSwitch` 需先做 2-choice 重编码（并剔除跨轴/跨维度错误），否则 2AFC DDM 解释不成立。
- 对条件极不均衡任务：应避免单被试分层拟合，优先层级回归形式，并在 posterior predictive 中检查 choice 概率是否合理。
- 对切换任务（DT/EmotionSwitch）：建议采用两个并行对照模型（Model A：Mixing；Model B：Switch），避免在同一模型中同时放入 `block` 与 `trial_type` 导致解释混淆；并在报告中明确 Model B 仅基于 mixed block。 
