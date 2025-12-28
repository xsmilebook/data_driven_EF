# EFNY 行为指标（Python）计算说明

本文档描述当前仓库中 `src/metric_compute/efny` 这套 **Python** 流程对 EFNY 行为数据（xlsx）进行读取、清洗、计算并导出行为指标宽表的方式。

## 1. 输入数据

### 1.1 输入文件

- 输入目录：`data/EFNY/behavior_data/cibr_app_data/`
- 输入文件格式：`*.xlsx`
- 每个被试通常对应一个文件，命名形如：`THU_20231014_131_ZXM_赵夕萌_GameData.xlsx`

### 1.2 输出文件

- 原始指标宽表：`data/EFNY/table/metrics/EFNY_beh_metrics.csv`
- 每行对应一个 xlsx 文件（一个被试）。
- 关键列：
  - `subject_code`：由文件名去掉 `_GameData.xlsx` 得到
  - `file_name`：文件名（basename）

## 2. xlsx → DataFrame：列名标准化

在读取每个 sheet 后，首先调用 `normalize_columns()`（见 `src/metric_compute/efny/io.py`）将中文列名统一为英文列名：

- `任务` → `task`
- `游戏序号` → `trial_index`
- `被试编号（用户账号）` → `subject_id`
- `被试姓名（真实姓名)` → `subject_name`
- `正式阶段刺激图片/Item名` → `item`
- `正式阶段正确答案` → `answer`
- `正式阶段被试按键` → `key`
- `绝对时间(待定)` → `abs_time`
- `相对时间(秒)` → `rt`

注：原始文件中还可能包含例如 `平均反应时(秒)`、`正确率` 等列，但当前计算主要使用上述标准化后的列。

## 3. Sheet 名 → 任务名映射

在 `run_raw()` 流程中（见 `src/metric_compute/efny/main.py`），会遍历每个 xlsx 的所有 sheet，并使用 `normalize_task_name(sheet_name)` 将 sheet 名归一化成内部任务名：

- 包含 `1back` 的 sheet：
  - 含 `number` → `oneback_number`
  - 含 `spatial` → `oneback_spatial`
  - 含 `emotion` → `oneback_emotion`
- 包含 `2back` 的 sheet：
  - 含 `number` → `twoback_number`
  - 含 `spatial` → `twoback_spatial`
  - 含 `emotion` → `twoback_emotion`
- 其他 sheet：默认保持原名（例如 `FLANKER`, `SST`, `DT`, `EmotionSwitch` 等）

> 说明：实际 xlsx sheet 常见命名如 `Number1Back/Spatial1Back/Emotion1Back` 等，这些会被 `.lower()` 后匹配到 `1back/2back` 分支。

## 4. 通用预处理（所有需要 RT 的任务）

多数任务在进入具体分析函数前会调用 `prepare_trials()`（见 `src/metric_compute/efny/preprocess.py`），该函数完成以下处理：

### 4.1 正确性标注：`correct_trial`

若数据中没有 `correct_trial` 列，则会基于 `answer` 与 `key` 计算：

- `correct_trial = (answer == key)`
- `answer` 缺失时 `correct_trial = NA`

内部实现使用 pandas nullable boolean 类型：可同时容纳 `True/False/NA`。

### 4.2 RT 转数值

- 将 `rt` 转为数值：`pd.to_numeric(..., errors='coerce')`
- 转换失败得到 `NaN`

### 4.3 RT 过滤与 3SD 剪裁（可配置）

在 `filter_rt=True` 时：

- 先按阈值过滤：保留 `rt_min <= rt <= rt_max`
  - 默认 `rt_min = 0.2`
  - `rt_max` 由各任务配置提供（例如多数任务 2.5 秒）
- 再做 3SD 剪裁：
  - 计算当前保留 RT 的均值 `m` 与标准差 `s`
  - 若 `s > 0`，保留 `| (rt - m) / s | <= 3`

### 4.4 最低保留比例 QC：`min_prop`

记：

- `n_raw`：原始试次数
- `n_problem`：
  - `rt` 缺失的 trial 数
  - + 在阈值过滤 / 3SD 剪裁中被删除的 trial 数

当 `n_problem > n_raw * (1 - min_prop)` 时（默认 `min_prop=0.5`），返回 `ok=False`，该任务整体指标会输出为缺失（NaN）。

> 换言之：要求“至少保留约 min_prop 的有效 trial”，否则认为该任务数据质量不足。

## 5. 输出字段命名规则

对每个被试文件的每个任务，输出列名统一为：

- `{task_name}_{metric_name}`

例如：

- `FLANKER_ACC`
- `oneback_number_dprime`
- `EmotionSwitch_Switch_Cost_RT`

## 6. 每个任务计算的指标（完整清单）

以下清单来自 `DEFAULT_TASK_CONFIG`（见 `src/metric_compute/efny/main.py`），即 **最终会写入 `EFNY_beh_metrics.csv` 的指标集合**。

### 6.1 N-back（6 个任务，指标相同）

任务：

- `oneback_number`
- `oneback_spatial`
- `oneback_emotion`
- `twoback_number`
- `twoback_spatial`
- `twoback_emotion`

指标：

- `ACC`：平均正确率（`correct_trial` 的均值）
- `RT_Mean`：正确试次的平均 RT
- `RT_SD`：正确试次 RT 的标准差（ddof=1, sample std）
- `Hit_Rate`：target 试次的正确率
- `FA_Rate`：nontarget 试次的 false alarm rate（= 在 nontarget 上错误的比例）
- `dprime`：`NormalDist().inv_cdf(H) - NormalDist().inv_cdf(FA)`，并使用 0.5/N 的边界校正

其中 target/nontarget 的定义：

- 将 `item` 与其前 `n_back` 个 trial 的 `item` 对比
- 相等 → `target`
- 不等（且两者均非空） → `nontarget`

### 6.2 冲突任务（FLANKER / ColorStroop / EmotionStroop）

每个任务指标相同：

- `ACC`
- `RT_Mean`
- `RT_SD`
- `Congruent_ACC`
- `Congruent_RT`
- `Incongruent_ACC`
- `Incongruent_RT`
- `Contrast_RT` = `Incongruent_RT - Congruent_RT`
- `Contrast_ACC` = `Incongruent_ACC - Congruent_ACC`

条件划分方式（见 `src/metric_compute/efny/metrics.py`）：

- FLANKER：依据 `item` 字符串后缀（如 `LL/RR` vs `LR/RL`）
- ColorStroop：解析 `item` 中 `pic_color` 与 `text_color` 是否一致
- EmotionStroop：从 `item` 抽取数字并按模/区间规则划分一致/不一致

### 6.3 切换任务（DCCS / DT / EmotionSwitch）

每个任务指标相同：

- `ACC`
- `RT_Mean`
- `RT_SD`
- `Repeat_ACC`
- `Repeat_RT`
- `Switch_ACC`
- `Switch_RT`
- `Switch_Cost_RT` = `Switch_RT - Repeat_RT`
- `Switch_Cost_ACC` = `Switch_ACC - Repeat_ACC`

Switch / Repeat 的定义（见 `analyze_switch()`）：

- 先基于任务生成 `rule`：
  - DCCS：`item` 的第 1 个字符
  - DT：`item` 含 `T/t` → `TN`，否则 `CN`
  - EmotionSwitch：从 `item` 提取数字
    - 1–4 → `emotion`
    - 5–8 → `gender`
- 将当前 `rule` 与上一 trial 的 `rule` 对比：
  - 相同 → `repeat`
  - 不同 → `switch`

混合阶段截断（Mixed block）：

- 若配置了 `mixed_from` 且存在 `trial_index`，则只保留 `trial_index >= mixed_from` 的 trial 参与计算。

### 6.4 SST

指标：

- `ACC`：Go trial 的正确率（来自 `prepare_trials(go_trials)`）
- `RT_Mean` / `RT_SD`：Go trial 正确试次 RT 均值/标准差
- `SSRT`：Integration method（从 Go RT 分布的分位数减去 Mean SSD）
- `Mean_SSD`：Stop trial 的平均 SSD（若 SSD 看起来是 ms，则除以 1000）
- `Stop_ACC`：Stop trial 抑制成功率（按 key 是否属于 no-press 别名判断）
- `Go_RT_Mean`, `Go_RT_SD`：同 `RT_Mean/RT_SD`

### 6.5 Go/NoGo（GNG 与 CPT）

指标：

- `ACC`
- `RT_Mean` / `RT_SD`：Go trial 正确试次 RT 统计
- `Go_ACC`
- `NoGo_ACC`
- `Go_RT_Mean`, `Go_RT_SD`
- `dprime`：由 Go 命中率与 NoGo false alarm 率计算（使用 `compute_dprime`）

其中 Go/NoGo 定义：

- `answer` 为 `true/1/yes` → Go
- `answer` 为 `false/0/no` → NoGo
- key 的 no-press 别名：`no/none/不按/未按/空/null/n/a/nan/''` 等

### 6.6 ZYST

指标：

- `ACC`
- `RT_Mean` / `RT_SD`
- `T0_ACC`
- `T1_ACC`
- `T1_given_T0_ACC`：仅在对应 trial 的 T0 正确时统计 T1 正确率
- `T0_RT`
- `T1_RT`

其中 trial/subtrial 的解析方式：

- 从 `trial_index` 解析形如 `trial-subtrial` 的结构（例如 `12-0`, `12-1`）

### 6.7 FZSS

指标：

- `ACC`
- `RT_Mean` / `RT_SD`
- `Overall_ACC`
- `Miss_Rate`
- `FA_Rate`
- `Correct_RT_Mean` / `Correct_RT_SD`

其中：

- `correct_trial = (answer == key)`（小写、去空格后比较）
- `Miss_Rate`：在 `answer == 'right'` 的 trial 中，`key != 'right'` 的比例
- `FA_Rate`：在 `answer == 'left'` 的 trial 中，`key == 'right'` 的比例

### 6.8 KT

指标：

- `ACC`
- `RT_Mean` / `RT_SD`
- `Overall_ACC`
- `Mean_RT`

说明：KT 当前等价于对经过 `prepare_trials()` 的数据做整体 ACC 与正确试次 RT 统计，并同时以两组字段名输出。

## 7. 关键参数（各任务的默认阈值）

见 `DEFAULT_TASK_CONFIG`（`src/metric_compute/efny/main.py`）：

- 多数任务：
  - `filter_rt = True`
  - `rt_max = 2.5`
  - `min_prop = 0.5`
- 特殊：
  - `FLANKER`: `rt_max = 2.0`
  - `FZSS`: `rt_max = 2.0`
  - `KT`: `rt_max = 3.0`
  - `GNG/CPT`: `filter_rt = False`（但内部仍会对 Go trial 做 `prepare_trials(filter_rt=True)`）

## 8. 运行入口

生成 `EFNY_beh_metrics.csv` 的入口函数：

- `src/metric_compute/efny/main.py: run_raw(data_dir, out_csv, task_config=None)`

它会：

- 遍历 `data_dir` 下所有 `.xlsx`
- 对每个文件调用 `process_file_raw()`
- 遍历每个 sheet → 归一化任务名 → 找到 task_config → 读入并标准化列 → `get_raw_metrics()` 计算指标
- 汇总为宽表并导出 CSV
