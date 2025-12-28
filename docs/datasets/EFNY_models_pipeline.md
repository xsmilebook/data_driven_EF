# src/models 主入口处理流程文档

本文档描述 `src/models/` 模块化“脑-行为关联分析（PLS / Sparse-CCA）”管线的**主入口与完整数据流**，用于帮助你快速定位：

- 数据从哪里加载、如何对齐
- 质量控制（缺失/低方差）与混杂回归（confound regression）怎么做
- PLS / Sparse-CCA 模型如何创建、拟合与评估
- 交叉验证（CV）与置换检验（Permutation test）的执行逻辑
- 结果如何保存与命名

## 1. 模块结构与职责

`src/models/` 目录下核心模块如下：

- **`__init__.py`**：包对外暴露 API（建议从这里 import）
- **`data_loader.py`**：EFNY 数据加载（脑连接矩阵/行为表/sublist）
- **`preprocessing.py`**：混杂变量回归（`ConfoundRegressor`）+ 数据质量过滤（`DataQualityFilter`）
- **`models.py`**：模型统一接口（PLS / Sparse-CCA / Adaptive-PLS）与工厂函数（`create_model`）
- **`evaluation.py`**：通用评估框架（`CrossValidator`、`PermutationTester`、`run_nested_cv_evaluation`）
- **`utils.py`**：日志、路径、配置（`ConfigManager`）、结果保存（`save_results`）
- **`config.json`**：默认配置文件（会被 `ConfigManager` 自动加载）
- **`example_usage.py`**：用法示例（合成数据/真实数据/保存结果）

## 2. 对外主入口（推荐 import 路径）

在多数脚本中建议按如下方式使用（由 `src/models/__init__.py` 暴露）：

- 数据加载：`EFNYDataLoader`, `create_synthetic_data`
- 预处理：`ConfoundRegressor`, `DataQualityFilter`, `create_preprocessing_pipeline`
- 建模：`create_model`, `PLSModel`, `SparseCCAModel`, `AdaptivePLSModel`
- 评估：`CrossValidator`, `PermutationTester`, `run_nested_cv_evaluation`
- 工具：`ConfigManager`, `setup_logging`, `save_results`, `get_results_dir`

## 3. 配置系统（ConfigManager + config.json）

### 3.1 默认配置加载规则

`ConfigManager` 在初始化时会：

- 先加载内置默认配置（`utils.py` 内 `_load_default_config()`）
- 再尝试加载 `src/models/config.json`（若存在）并覆盖默认值
- 若你显式传入 `ConfigManager(config_file=...)`，会继续加载该配置并覆盖

### 3.2 config.json 关键字段（与你的主流程强相关）

- **`data.root_dir`**：数据根目录（EFNY 数据集根）
- **`data.brain_file`**：脑连接矩阵 `.npy` 相对路径
- **`data.behavioral_file`**：行为表 `.csv` 相对路径
- **`data.sublist_file`**：被试列表 `sublist.txt`
- **`preprocessing.confound_variables`**：混杂变量列名（默认 `sex/age/meanFD`）
- **`behavioral.selected_measures`**：选择进入模型的行为指标列名
- **`evaluation.cv.*`**：CV 配置（折数、shuffle、random_state、是否 stratify）
- **`evaluation.permutation_test.*`**：置换检验配置（次数、随机种子）
- **`output.*`**：结果保存目录、命名、是否创建 timestamp 子目录等

> 注意：`src/models` 模块提供的是“库 + 管线组件”。具体的命令行入口/批处理（HPC）通常在 `src/scripts` 里完成，并会读取/覆盖这些配置。

## 4. 数据输入与对齐（非常关键）

### 4.1 EFNYDataLoader 的输入文件

`EFNYDataLoader`（`data_loader.py`）默认读取：

- `fc_vector/Schaefer100/EFNY_Schaefer100_FC_matrix.npy`
- `table/demo/EFNY_behavioral_data.csv`
- `table/sublist/sublist.txt`

并提供：

- `load_brain_data() -> np.ndarray`：形状 `(n_subjects, n_features)`
- `load_behavioral_data() -> pd.DataFrame`：形状 `(n_subjects, n_measures)`
- `load_subject_list() -> np.ndarray[str]`：形状 `(n_subjects,)`
- `load_all_data() -> (brain_data, behavioral_data, subject_ids)`

### 4.2 对齐假设

当前 `load_all_data()` 只检查三者 **行数一致**，并**不做基于 subject_id 的重排/merge**。

因此隐含假设是：

- `brain_data[i]`、`behavioral_data.iloc[i]`、`subject_ids[i]` 对应同一被试

如果你的行为表在预处理或合并后发生了行顺序改变，必须在进入模型前自行对齐（例如按 subject_id merge/reindex）。

## 5. 典型处理流程（真实数据：从加载到保存）

下面按“你在主脚本里一般会怎么串联”来描述处理流程。

### 5.1 初始化日志与配置

- 调用 `setup_logging()` 统一设置日志输出（控制台 + 可选文件）
- 使用 `ConfigManager()` 读取并合并配置

输出：

- 全局 logger 已配置
- `config` 字典可用于驱动后续步骤

### 5.2 加载数据

- `EFNYDataLoader(data_root=...)`
- `brain_data, behavioral_df, subject_ids = load_all_data()`

得到：

- **X（脑数据）**：通常是 `np.ndarray`
- **行为表（包含候选 Y 与协变量）**：`pd.DataFrame`
- **subject_ids**：`np.ndarray[str]`

### 5.3 选择行为指标（构建 Y）

根据配置 `behavioral.selected_measures` 从 `behavioral_df` 中选择列构建 `Y`。

推荐做法（主脚本侧完成）：

- 过滤掉缺失列（避免 `KeyError`）
- 保留列顺序与配置一致（便于可重复性）

### 5.4 构建协变量 confounds

`src/models` 的 `ConfoundRegressor` 与 `CrossValidator` 都需要你提供 `confounds`：形状 `(n_subjects, n_confounds)`。

常见来源：

- 直接从 `behavioral_df` 中取 `['sex', 'age', 'meanFD']`

### 5.5 数据质量控制（可选）

你可以在进入模型前做质量控制（缺失率、低方差等），相关组件：

- `DataQualityFilter(max_missing_rate=..., min_std=...)`

典型策略：

- **按特征过滤**：剔除缺失率高或几乎常数的特征
- **按被试过滤**：剔除缺失率过高的被试

> 注意：过滤会改变样本数/特征数，且必须对 X/Y/confounds 同步应用相同的被试掩码。

### 5.6 混杂变量回归 + 残差标准化

使用 `ConfoundRegressor(standardize=True)`：

- `fit()` 在训练集上对每个特征拟合 `LinearRegression(confounds -> feature)`
- `transform()` 得到残差 `feature - predicted(confounds)`
- 若 `standardize=True`，还会对残差做 z-score（使用训练集残差均值/方差）

得到：

- `X_clean`, `Y_clean`

### 5.7 创建模型（PLS / Sparse-CCA / rCCA / Adaptive）

使用 `create_model(model_type, **kwargs)`（`models.py`）：

- `model_type='pls'`：`PLSModel`，底层是 `sklearn.cross_decomposition.PLSCanonical`
- `model_type='scca'`：`SparseCCAModel`，底层是 `cca-zoo` 的 `SCCA_IPLS`
- `model_type='adaptive_pls'`：`AdaptivePLSModel`，内部 CV 选择成分数
- `model_type='rcca'`：`RCCAModel`，底层是 `cca-zoo` 的 `rCCA`（正则化 CCA）
- `model_type='adaptive_scca'`：`AdaptiveSCCAModel`，内部 CV 同时选择 `n_components` 与 `(sparsity_X, sparsity_Y)`
- `model_type='adaptive_rcca'`：`AdaptiveRCCAModel`，内部 CV 同时选择 `n_components` 与 `(c_X, c_Y)`

关键参数：

- `n_components`：成分数量（或自适应范围）
- `scale`：PLS 内部是否标准化（注意与外部 `ConfoundRegressor` 的标准化关系）

对 rCCA 额外关键参数：

- `c_X`, `c_Y`：两视图正则化强度（对应 `cca_zoo.linear.rCCA(c=[c_X, c_Y])`）
- `pca`：是否启用 PCA 求解优化（高维 FC 推荐 `True`）
- `eps`：数值稳定参数

### 5.8 拟合与核心输出

对清理后的数据拟合模型：

- `model.fit(X_clean, Y_clean)`

常用的后续输出：

- `X_scores, Y_scores = model.transform(X_clean, Y_clean)`
- `canonical_corrs = model.calculate_canonical_correlations(X_scores, Y_scores)`
- `X_loadings, Y_loadings = model.get_loadings()`
- `variance = model.calculate_variance_explained(X_clean, Y_clean, X_scores, Y_scores)`（PLS 有实现；SCCA 的解释方式可能需要单独定义）

### 5.9 交叉验证（CrossValidator）

`CrossValidator.run_cv_evaluation(model, X, Y, confounds=...)` 在每一折中执行：

1. 划分训练/测试索引
2. 在训练集拟合混杂回归
3. 将训练集与测试集转换为“去混杂+标准化”的空间
4. 在训练集拟合模型
5. 在测试集计算 scores 与 canonical correlations

可选：

- `stratify=True`：使用 `KMeans` 对 Y 聚类得到伪标签，再用 `StratifiedKFold` 分层切分

**重要提示（实现细节）**：

- 当前实现里，若 `confounds` 不为空，CV 每折会实例化 `ConfoundRegressor(standardize=True)`。
- 你在复用/扩展该 CV 框架时，应确保 X 与 Y 的混杂回归不会相互覆盖（建议为 X 与 Y 分别使用不同的回归器实例）。

### 5.10 置换检验（PermutationTester）

`PermutationTester.run_permutation_test(model, X, Y, confounds=..., permutation_seed=...)` 执行：

1. 对 Y 行进行随机置换（打乱被试-行为配对）
2. （可选）对置换后的 Y 与 X 做混杂回归
3. 拟合模型并计算 canonical correlations

然后你可以聚合多次置换得到：

- `permuted_correlations`：形状 `(n_perm, n_components)`
- `p_values = calculate_p_values(observed_correlations, permuted_correlations)`

**Adaptive 模型在置换检验中的策略（重要）**：

- `adaptive_pls`：真实数据（`task_id=0`）先选出最优 `n_components`；置换检验（`task_id>0`）固定该 `n_components`，改用标准 `pls`。
- `adaptive_scca`：真实数据先选出最优 `(n_components, sparsity_X, sparsity_Y)`；置换检验固定该组合，改用标准 `scca`。
- `adaptive_rcca`：真实数据先选出最优 `(n_components, c_X, c_Y)`；置换检验固定该组合，改用标准 `rcca`。

这样可避免“每次置换都重新调参”导致的统计量不可比与潜在信息泄露。

### 5.11 保存结果（save_results）

`save_results(results_dict, output_path, format='both')` 会输出：

- `output_path.json`：可读的 JSON（会把 numpy 数组转换成 list）
- `output_path.npz`：压缩的二进制数组（更适合大矩阵，例如 loadings）

建议在 `results_dict` 中至少包含：

- 数据形状与特征数量、被试数
- 关键配置（模型类型、成分数、selected_measures、confounds 列）
- 真实拟合的 canonical correlations、loadings
- CV 汇总结果（均值/标准差）
- 置换检验分布与 p 值（若启用）

## 6. 与 src/scripts 的关系（你项目中“真正可跑”的主入口通常在 scripts）

你的项目里：

- `src/models/` 更偏“可复用组件与框架”
- `src/scripts/` 往往提供：
  - 命令行参数解析
  - 读取/覆盖配置
  - 选择不同 atlas/数据子集
  - HPC 任务拆分（真实任务 + permutations）
  - 结果命名、落盘与汇总

因此，当你要追踪“从命令行到最终结果文件”的全链路时，通常需要同时看：

- `src/scripts/*`（主入口与批处理）
- `src/models/*`（核心算法与评估逻辑）

