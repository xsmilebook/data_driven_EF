# 方法（Methods）

本文件总结当前交叉验证与置换检验流程的实现细节，参考：

- `scripts/run_single_task.py`
- `src/models/evaluation.py`
- `configs/analysis.yaml`

以下描述强调实现一致性，不包含结果性结论。

## 嵌套交叉验证（真实数据）

入口：`scripts/run_single_task.py`（`task_id=0`）调用 `src/models/evaluation.py` 中的 `run_nested_cv_evaluation`。

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
- 若模型提供，则包含加载（loadings）。

汇总输出包括：

- `outer_mean_canonical_correlations` 与 `outer_std_canonical_correlations`
- 外层测试相关矩阵（用于后续汇总）

结果保存路径：

```
outputs/<DATASET>/results/real/<atlas>/<model_type>/seed_<seed>/
```

## 置换检验（逐成分）

当 `task_id > 0` 时，`scripts/run_single_task.py` 进入置换流程。

### 1) 置换种子与打乱方式

- `permutation_seed = random_state + task_id`
- 对 Y 做被试行打乱，X 保持不变。

### 2) 真实数据摘要的依赖

置换检验依赖真实数据的逐成分摘要，由以下脚本生成：

- `src/result_summary/summarize_real_loadings_scores.py`

读取路径：

```
outputs/<DATASET>/results/summary/<atlas>/<model_type>/
```

摘要包含真实数据的逐成分得分与被试得分，用于后续逐成分置换。

### 3) 逐成分置换流程

令 `n_components` 为真实摘要中的成分数量。对每个 `k = 1..n_components`：

- 构造附加协变量：使用真实数据的被试得分中 `1..k-1` 成分：
  - `X_scores[:, :k-1]` 与 `Y_scores[:, :k-1]`
  - 若存在原始协变量，则与其拼接。
- 构建参数网格并强制 `n_components=1`。
- 使用 `run_nested_cv_evaluation` 在 `Y_perm` 上执行嵌套 CV。
- 取 `outer_mean_canonical_correlations` 的第一个元素作为该 k 的置换得分。

置换结果保存每个 k 的得分与嵌套 CV 细节。

### 4) 输出位置

置换结果保存路径：

```
outputs/<DATASET>/results/perm/<atlas>/<model_type>/seed_<seed>/
```

元数据包含置换种子与 `configs/analysis.yaml` 中的 `permutation_n_iters`（用于可追溯性记录）。
