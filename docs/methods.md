# 方法（Methods）

本文档总结当前项目的方法实现细节。以下描述强调实现一致性，不包含结果性结论。

## 预处理方法学细节

本节补充影像与行为预处理的关键方法学假设，确保实现与评估一致。

### neuroimgaing data preprocess


#### Functional connectivity


#### head motion qc


### behavioral data preprocess

本节定义 THU `app_data` 行为任务预处理的 v1 实现规格。该规格用于后续实现
`scripts/behavior/app_check_format.py`、`scripts/behavior/app_clean.py` 与
`scripts/behavior/app_metrics.py`；本节不要求生成运行时结果。

#### 数据范围

- 输入范围：`data/raw/THU/app_data/*.xlsx`。
- 数据单位：每个 workbook 视为 1 名被试，每个 sheet 视为 1 个任务表。
- `subject_code`：由 workbook 文件名移除 `_GameData.xlsx` 后得到。
- v1 支持当前观察到的 18 类 app 任务 sheet：
  `FLANKER`、`SST`、`FZSS`、`DT`、`ColorStroop`、`EmotionStroop`、`CPT`、
  `EmotionSwitch`、`Number1Back`、`Number2Back`、`Spatial1Back`、`Spatial2Back`、
  `Emotion1Back`、`Emotion2Back`、`KT`、`ZYST`、`DCCS`、`GNG`。
- XY、BNU、inventory、demography 与 task-fMRI 行为日志不纳入本 v1 口径，后续单独补充。

#### 字段标准化

原始中文列名在读入后映射为稳定英文列名。标准字段如下：

- `任务` -> `task`
- `游戏序号` -> `trial_index`
- `被试编号（用户账号）` -> `subject_id`
- `被试姓名（真实姓名)` -> `name`
- `正式阶段刺激图片/Item名` -> `item`
- `正式阶段正确答案` -> `answer`
- `正式阶段被试按键` -> `key`
- `相对时间(秒)` -> `rt`
- `SSRT` -> `ssrt_or_ssd`
- `空屏时长`、`全部时间(秒)`、`crossPicDur` 在 CPT 中保留为任务特异辅助字段。

Excel 原始列中的 `平均反应时(秒)` 与 `正确率` 仅作为原始记录保留，不作为最终
`RT_Mean` 或 `ACC` 的来源。最终指标必须由标准化试次长表重新计算。

#### 格式检查

`check_format` 阶段只检查原始表结构，不改变输入文件：

- 检查每个 workbook 是否至少包含 1 个受支持任务 sheet。
- 检查通用必需列：`任务`、`游戏序号`、`被试编号（用户账号）`、`正式阶段正确答案`、
  `正式阶段被试按键`、`相对时间(秒)`。
- 检查任务特异列：SST 必须包含 `SSRT`；CPT 若存在 `空屏时长`、`全部时间(秒)`、
  `crossPicDur` 则原样保留。
- 对缺失关键列、空 sheet、未知 sheet 名、重复 `subject_code`-`task` 组合生成 QC 记录。
- 格式检查不因单个任务失败而中断全体扫描；失败任务在 QC 表中标记，后续清洗跳过。

#### 试次清洗与排除

清洗以 `subject_code`-`task` 为单位执行，默认参数来自未来
`configs/behavioral_metrics.yaml`：

- `rt_min=0.2`
- `rt_max=10`
- `rt_outlier_sd=3`
- `min_valid_prop=0.5`

通用清洗规则：

- `rt` 使用数值转换，非数值记为缺失。
- `correct_trial` 由 `answer == key` 派生；比较前去除首尾空白，缺失值不视为相等。
- `valid_for_acc` 表示该试次可用于准确率或抑制成功率计算。
- `valid_for_rt` 表示该试次可用于 RT 指标计算。
- `valid_for_acc` 与 `valid_for_rt` 分开维护。SST stop 试次、GNG/CPT nogo 试次等允许无反应的试次，不因 `rt` 缺失被排除出准确率计算。
- RT 指标只使用满足 `rt_min <= rt <= rt_max` 的试次，并在任务内进一步排除
  `rt > mean + rt_outlier_sd * sd` 的高值试次。
- 任务内 `valid_prop = n_valid_acc / n_trials_raw`；若 `valid_prop < min_valid_prop`，
  该任务标记为 `ok=False`，指标输出为缺失或不输出，并在 `qc_reason` 中记录原因。

#### 指标计算与输出

所有指标均基于清洗后的标准化长表计算。比例指标在分母为 0 时记为缺失；`dprime`
计算需对 0 或 1 的命中率/虚报率做有限样本校正。

1. N-back（`Number1Back`、`Number2Back`、`Spatial1Back`、`Spatial2Back`、
   `Emotion1Back`、`Emotion2Back`）

- `n_back` 由任务名中的 `1Back` 或 `2Back` 确定。
- 以 `item` 序列定义 `target`：当前 `item` 与前 `n_back` 个试次的 `item` 相同。
- 前 `n_back` 个无法判定 target 的试次不进入 `Hit_Rate` 与 `FA_Rate` 分母。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`Hit_Rate`、`FA_Rate`、`dprime`。

2. 冲突任务（`FLANKER`、`ColorStroop`、`EmotionStroop`）

- `FLANKER`：由 `item` 中末尾方向组合判定一致/不一致，如 `LL`、`RR` 为
  `congruent`，`LR`、`RL` 为 `incongruent`。
- `ColorStroop`：解析 `Pic_<color>_Text_<color>`，图片颜色与文字颜色相同为
  `congruent`，不同为 `incongruent`。
- `EmotionStroop`：按任务规则将 `item` 或答案编码映射到情绪一致/不一致条件；若无法解析，记录为条件缺失，不进入条件指标分母。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`Congruent_ACC`、`Congruent_RT`、
  `Incongruent_ACC`、`Incongruent_RT`、`Contrast_RT`、`Contrast_ACC`。
- `Contrast_RT = Incongruent_RT - Congruent_RT`；`Contrast_ACC = Incongruent_ACC - Congruent_ACC`。

3. 任务切换（`DCCS`、`DT`、`EmotionSwitch`）

- `DCCS`：以 `item` 首字符代表规则。
- `DT`：以 `item` 中的任务线索映射规则；当前 THU app 记录中优先按既定任务规则解析
  `CN`/`TN` 或等价编码。
- `EmotionSwitch`：以 `item` 中的规则线索映射 emotion/gender 规则。
- 只在 mixed block 计算 switch/repeat 指标：DCCS 从 trial 21 开始；DT 与
  EmotionSwitch 从 trial 65 开始。
- 当前试次规则与上一可解析试次规则不同为 `switch`，相同为 `repeat`；mixed block
  首个可解析试次不进入 switch/repeat 分母。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`Repeat_ACC`、`Repeat_RT`、`Switch_ACC`、
  `Switch_RT`、`Switch_Cost_RT`、`Switch_Cost_ACC`。
- `Switch_Cost_RT = Switch_RT - Repeat_RT`；`Switch_Cost_ACC = Switch_ACC - Repeat_ACC`。

4. 停止信号任务（`SST`）

- `ssrt_or_ssd` 有数值的试次视为 stop 试次；缺失或 `-` 视为 go 试次。
- stop 试次正确性为未按键；go 试次正确性为 `answer == key`。
- go RT 指标仅使用正确 go 试次，并应用 RT 范围与 3 SD 修剪。
- 输出：`ACC`、`Stop_ACC`、`Mean_SSD`、`SSRT`、`Go_RT_Mean`、`Go_RT_SD`。
- `SSRT` 采用积分法：`SSRT = Go_RT_quantile(p) - Mean_SSD`，其中
  `p = 1 - Stop_ACC`。

5. Go/No-Go 与 CPT（`GNG`、`CPT`）

- `answer` 为 `true`、`yes`、`1` 或等价真值时视为 go；`false`、`no`、`0`
  或等价假值时视为 nogo。
- go 正确为按键符合目标反应；nogo 正确为无反应或记录为抑制成功。
- RT 指标仅使用正确 go 试次，并应用 RT 范围与 3 SD 修剪；nogo 试次不进入 RT 分母。
- 输出：`ACC`、`Go_ACC`、`NoGo_ACC`、`Go_RT_Mean`、`Go_RT_SD`、`dprime`。
- `dprime` 中 `Go_ACC` 作为命中率，`1 - NoGo_ACC` 作为虚报率。

6. ZYST（`ZYST`）

- `trial_index` 解析为 `(trial, subtrial)`，仅保留包含 0 与 1 两个子试次的 trial
  进入条件化指标。
- `T1_given_T0_ACC` 只在 T0 正确的 trial 中计算 T1 正确率。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`T0_ACC`、`T1_ACC`、`T1_given_T0_ACC`、
  `T0_RT`、`T1_RT`。

7. FZSS（`FZSS`）

- 正确性由 `answer == key` 定义。
- `Miss_Rate`：在 `answer == right` 的试次中，`key != right` 的比例。
- `FA_Rate`：在 `answer == left` 的试次中，`key == right` 的比例。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`Miss_Rate`、`FA_Rate`、
  `Correct_RT_Mean`、`Correct_RT_SD`。

8. KT（`KT`）

- 正确性由 `answer == key` 定义。
- 输出：`ACC`、`RT_Mean`、`RT_SD`、`Overall_ACC`、`Mean_RT`。
- `Overall_ACC` 与 `Mean_RT` 分别与 `ACC`、`RT_Mean` 等价，用于下游兼容。

目标输出表：

- 清洗长表：`dataset`、`subject_code`、`task`、`trial_index`、`item`、`answer`、
  `key`、`rt`、`correct_trial`、`valid_for_acc`、`valid_for_rt`、`exclusion_reason`。
- 任务 QC 表：`subject_code`、`task`、`n_trials_raw`、`n_valid_acc`、`n_valid_rt`、
  `valid_prop`、`ok`、`qc_reason`。
- 指标长表：`subject_code`、`task`、`metric`、`value`。
- 指标宽表：以 `subject_code` 为行，将 `task_metric` 展开为列，用于下游脑-行为关联分析。

建议输出目录为 `data/processed/THU/behavioral_metrics/`。该目录属于运行产物区域，
不纳入版本控制。

验证口径：

- 使用 1-3 个 THU workbook 做非持久化冒烟验证，确认 18 类任务 sheet 可被识别。
- 检查关键中文列存在，SST/CPT/GNG 的无反应试次不会因 RT 缺失被错误排除出准确率分母。
- 对 N-back target、Flanker/Stroop congruent、switch/repeat、SST stop/go、
  GNG/CPT go/nogo 各抽样核对至少 1 个试次。
- 文档或实现修改后执行 `git diff --check`；若只修改方法文档，不运行完整预处理。
- 若临时验证写入 `temp/` 或 `tmp/`，验证完成后立即清理。

## 嵌套交叉验证（真实数据）

### 1) 数据输入与配置

- 输入：脑影像特征 X、行为特征 Y、可选协变量 C。
- CV 默认值来自 `configs/analysis.yaml`，可由命令行覆盖：
  - `evaluation.cv_n_splits`, `evaluation.inner_cv_splits`
  - `evaluation.cv_shuffle`, `evaluation.inner_shuffle`
  - `evaluation.outer_shuffle_random_state`, `evaluation.inner_shuffle_random_state`
  - `evaluation.score_metric`（当前为 `mean_canonical_correlation`）
- 模型超参数候选由 `scripts/run_single_task.py` 根据模型类型（adaptive PLS / sCCA / rCCA）构建，并作为参数网格传入嵌套 CV。

### 2) 外层 CV

- 使用 `KFold(n_splits=outer_cv_splits, shuffle=outer_shuffle, random_state=outer_random_state)`。
- 对每个外层折：
  - 内层 CV 仅在外层训练集上选择超参数。
  - 用最优参数在完整外层训练集上重训。
  - 在外层测试集上评估。

### 3) 内层 CV（超参数选择）

- 使用 `KFold(n_splits=inner_cv_splits, shuffle=inner_shuffle, random_state=inner_random_state)`，数据来自外层训练集。
- 对每个参数候选：
  - 在内层训练集拟合模型。
  - 在内层验证集计算评分。
  - 以各内层折评分均值选择最优参数。
- 评分指标为各成分典型相关的均值（`mean_canonical_correlation`）。

### 4) 折内预处理（仅用训练拟合）

所有预处理均在训练集拟合，并应用于对应验证/测试集：

- 缺失值处理：按训练集特征均值填补 NaN。
- 协变量回归：`ConfoundRegressor(standardize=True)` 在训练集拟合并应用到验证/测试集。
- 可选标准化：对 X、Y 各自使用 `StandardScaler`。
- 可选 PCA：对 X、Y 分别 `PCA(n_components=...)`，仅用训练集拟合。

这些步骤在 `run_nested_cv_evaluation` 内部，对内层与外层折分别执行。

### 5) 嵌套 CV 输出

`run_nested_cv_evaluation` 输出包含：

- 外层折的最优参数与内层 CV 统计。
- 外层测试集典型相关及其均值。
- 外层训练集典型相关（用于诊断）。
- 若模型提供，则包含loadings。

汇总输出包括：

- `outer_mean_canonical_correlations` 与 `outer_std_canonical_correlations`
- 外层测试相关矩阵（用于后续汇总）

## 置换检验（逐成分）

### 1) 置换种子与打乱方式

- `permutation_seed = random_state + task_id`
- 对 Y 做被试行打乱，X 保持不变。

### 2) 逐成分置换流程

令 `n_components` 为真实摘要中的成分数量。对每个 `k = 1..n_components`：

- 构造附加协变量：使用真实数据的被试得分中 `1..k-1` 成分：
  - `X_scores[:, :k-1]` 与 `Y_scores[:, :k-1]`
  - 若存在原始协变量，则与其拼接。
- 构建参数网格并强制 `n_components=1`。
- 使用 `run_nested_cv_evaluation` 在 `Y_perm` 上执行嵌套 CV。
- 取 `outer_mean_canonical_correlations` 的第一个元素作为该 k 的置换得分。

置换结果保存每个 k 的得分与嵌套 CV 细节。

