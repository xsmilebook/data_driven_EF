# 数据驱动执行功能研究代码库文档

本代码库包含用于数据驱动执行功能（Executive Function, EF）研究的分析工具和脚本。主要分为四个功能模块：预处理、功能连接分析、行为指标计算和脑-行为关联分析。

## 目录结构

```
src/
├── preprocess/          # 数据预处理模块
├── functional_conn/     # 功能连接分析模块
├── metric_compute/      # 行为指标计算模块
├── models/              # 脑-行为关联分析模块（PLS/Sparse-CCA）
└── scripts/             # 脚本和HPC任务提交
```

## 预处理模块 (preprocess/)

### 1. `get_mri_sublist.py`
**功能**: 获取有效的MRI子列表
- 筛选符合特定标准的被试数据
- 生成用于后续分析的有效被试列表
- 输出被试数量统计信息

### 2. `screen_head_motion_efny.py`
**功能**: 头动筛查
- 检查MRI数据中的头动参数
- 根据预设标准排除头动过大的被试
- 确保数据质量符合分析要求

### 3. `generate_valid_sublists.py`
**功能**: 生成有效子列表
- 基于多种标准创建数据子集
- 为不同分析步骤准备相应的数据列表
- 管理数据筛选流程

### 4. `format_app_data.py`
**功能**: 格式化应用数据
- 将原始数据转换为标准格式
- 统一数据结构和命名规范
- 为后续分析准备输入数据

### 5. `batch_run_xcpd.sh`
**功能**: 批量运行XCP-D处理
- 自动化批量处理fMRI数据
- 调用XCP-D工具进行数据预处理
- 提高处理效率的批处理脚本

### 6. `xcpd_36p.sh`
**功能**: XCP-D 36参数处理
- 使用36参数模型处理fMRI数据
- 进行详细的噪声回归处理
- 生成高质量的时间序列数据

## 功能连接分析模块 (functional_conn/)

### 1. `compute_fc_schaefer.py`
**功能**: 基于Schaefer图谱计算功能连接
- 使用Schaefer脑图谱划分脑区
- 计算脑区之间的功能连接强度
- 生成功能连接矩阵

### 2. `compute_group_avg_fc.py`
**功能**: 计算组平均功能连接
- 计算被试组的功能连接平均值
- 生成组水平的连接矩阵
- 用于组间比较和统计分析

### 3. `fisher_z_fc.py`
**功能**: Fisher Z变换功能连接
- 对功能连接值进行Fisher Z变换
- 使数据更符合正态分布假设
- 提高统计分析的准确性

### 4. `plot_fc_matrix.py`
**功能**: 绘制功能连接矩阵
- 可视化功能连接矩阵
- 生成高质量的矩阵热图
- 支持自定义配色和标注

### 5. `submit_compute_fc.sh`
**功能**: 提交功能连接计算任务
- 在集群环境中提交计算作业
- 管理计算资源和任务调度
- 批量处理多个被试数据

### 6. `submit_fisher_z.sh`
**功能**: 提交Fisher Z变换任务
- 提交Fisher Z变换作业到计算集群
- 优化计算资源使用
- 自动化批量处理流程

## 行为指标计算模块 (metric_compute/)

### 1. `compute_efny_metrics.py`
**功能**: 计算EFNY行为指标
- 处理执行功能任务的行为数据
- 计算各项认知任务的性能指标
- 生成标准化的行为测量结果

### 2. `metrics_similarity_heatmap.py`
**功能**: 生成指标相似性热图
- 计算不同行为指标之间的相关性
- 生成指标相似性矩阵热图
- 可视化认知任务间的关联模式
- **修复内容**: 修正了Flanker任务指标显示问题，现在包含所有32个有效指标

## 脑-行为关联分析模块 (models/)

### 🚀 功能特性

- **模块化架构**: 数据加载、预处理、建模和评估的清晰分离
- **HPC就绪**: 通过SLURM作业数组支持并行置换检验
- **Sklearn兼容**: 遵循scikit-learn转换器模式，无缝集成
- **多模型支持**: PLS和Sparse-CCA统一接口
- **鲁棒预处理**: 支持交叉验证的混杂回归
- **全面评估**: 交叉验证、置换检验和嵌套CV
- **类型提示**: 完整的类型注解支持，更好的IDE集成

### 📁 模块结构

```
src/models/
├── __init__.py              # 包初始化
├── data_loader.py           # 数据加载工具
├── preprocessing.py         # 混杂回归和质量过滤
├── models.py               # PLS和Sparse-CCA模型实现
├── evaluation.py           # 交叉验证和置换检验
├── utils.py                # 日志、配置和工具函数
├── config.json             # 配置文件
└── example_usage.py        # 使用示例和演示
```

### 📊 快速开始

#### 基础分析

```python
from src.models import EFNYDataLoader, create_model, ConfoundRegressor

# 加载数据
data_loader = EFNYDataLoader()
brain_data, behavioral_data, subject_ids = data_loader.load_all_data()

# 创建合成协变量（或加载真实协变量）
covariates = pd.DataFrame({
    'sex': np.random.choice([0, 1], size=len(subject_ids)),
    'age': np.random.normal(25, 5, size=len(subject_ids)),
    'meanFD': np.random.normal(0.15, 0.05, size=len(subject_ids))
})

# 预处理：回归混杂因素
confound_regressor = ConfoundRegressor(standardize=True)
brain_clean = confound_regressor.fit_transform(brain_data, confounds=covariates)
behavioral_clean = confound_regressor.fit_transform(behavioral_data, confounds=covariates)

# 创建并拟合PLS模型
pls_model = create_model('pls', n_components=5, random_state=42)
pls_model.fit(brain_clean, behavioral_clean)

# 获取结果
X_scores, Y_scores = pls_model.transform(brain_clean, behavioral_clean)
canonical_corrs = pls_model.calculate_canonical_correlations(X_scores, Y_scores)
```

#### 自适应PLS模型（自动选择n_components）

```python
from src.models import create_model

# 创建自适应PLS模型 - 自动选择最优成分数量
adaptive_pls_model = create_model(
    'adaptive_pls',
    n_components_range=[1, 2, 3, 4, 5, 6],  # 搜索范围
    cv_folds=5,                            # 内部交叉验证折数
    criterion='canonical_correlation',     # 选择标准
    random_state=42
)

# 拟合模型（会自动选择最优n_components）
adaptive_pls_model.fit(brain_clean, behavioral_clean)

# 获取最优成分数量
optimal_n_components = adaptive_pls_model.optimal_n_components
print(f"自动选择的最优成分数量: {optimal_n_components}")

# 获取详细的交叉验证结果
cv_results = adaptive_pls_model.get_cv_results()
print("各成分数量评估结果:")
for n_comp, metrics in cv_results.items():
    print(f"  n_components={n_comp}: 典型相关={metrics['canonical_correlation']:.4f}")

# 使用模型进行预测
X_scores, Y_scores = adaptive_pls_model.transform(brain_clean, behavioral_clean)
canonical_corrs = adaptive_pls_model.calculate_canonical_correlations(X_scores, Y_scores)
```

#### 交叉验证

```python
from src.models import CrossValidator

# 创建交叉验证器
cv = CrossValidator(n_splits=5, shuffle=True, random_state=42)

# 运行交叉验证
cv_results = cv.run_cv_evaluation(pls_model, brain_clean, behavioral_clean)

# 获取汇总表
summary_df = cv.create_cv_summary_table(cv_results)
print(summary_df)
```

#### 置换检验

```python
from src.models import PermutationTester

# 创建置换检验器
perm_tester = PermutationTester(n_permutations=1000, random_state=42)

# 运行单次置换检验（用于HPC）
perm_result = perm_tester.run_permutation_test(
    pls_model, brain_clean, behavioral_clean, 
    permutation_seed=123
)

# 计算p值
p_values = perm_tester.calculate_p_values(real_correlations, permuted_correlations)
```

### 🏭 HPC使用

#### 单任务执行

```bash
# 真实数据分析
python src/scripts/run_single_task.py \
    --task_id 0 \
    --model_type pls \
    --n_components 5

# 自适应PLS模型（自动选择n_components）
python src/scripts/run_single_task.py \
    --task_id 0 \
    --model_type adaptive_pls \
    --n_components 6  # 最大搜索范围

# 置换检验（task_id = 1-1000 用于不同的置换）
python src/scripts/run_single_task.py \
    --task_id 1 \
    --model_type pls \
    --n_components 5
```

#### SLURM作业数组

```bash
# 提交数组作业进行置换检验
sbatch src/scripts/submit_hpc_job.sh

# 或提交特定范围
sbatch --array=1-1000 src/scripts/submit_hpc_job.sh
```

#### 命令行选项

```bash
python src/scripts/run_single_task.py --help

# 关键参数：
# --task_id: 0 表示真实数据，1-N 表示置换
# --model_type: pls, adaptive_pls 或 scca
# --n_components: 成分数量（对于adaptive_pls是最大搜索范围）
# --use_synthetic: 使用合成数据进行测试
# --regress_confounds: 是否回归混杂因素
# --run_cv: 是否运行交叉验证
# --cv_n_splits: CV折数
# --output_dir: 输出目录
# --log_level: 日志级别
```

### 🧪 测试

#### 运行示例

```bash
# 运行所有示例
python src/models/example_usage.py

# 运行特定示例
python -c "from src.models.example_usage import example_basic_analysis; example_basic_analysis()"
```

#### 合成数据测试

```bash
# 使用合成数据测试自适应PLS模型
python src/scripts/run_single_task.py \
    --task_id 0 \
    --model_type adaptive_pls \
    --n_components 5 \
    --use_synthetic \
    --n_subjects 100 \
    --n_brain_features 200 \
    --n_behavioral_measures 15
```

### 📈 输出格式

结果以JSON和NPZ格式保存：

#### JSON格式（人类可读）
```json
{
  "task_type": "real_data",
  "task_id": 0,
  "model_info": {
    "model_type": "PLS",
    "n_components": 5
  },
  "canonical_correlations": [0.65, 0.42, 0.28, 0.15, 0.08],
  "variance_explained_X": [8.5, 12.3, 15.1, 17.2, 19.0],
  "variance_explained_Y": [22.1, 35.6, 42.8, 48.2, 52.1],
  "metadata": {
    "timestamp": "20241214_143000",
    "n_samples": 394,
    "n_features_X": 4950,
    "n_features_Y": 30
  }
}
```

#### NPZ格式（高效存储）
- 包含分数、载荷和其他数值数据的numpy数组
- 压缩存储，节省空间
- 易于加载进行进一步分析

### 🔍 模型比较

| 模型 | 描述 | 使用场景 | 实现状态 | 特点 |
|-------|-------------|----------|---------------------|-------|
| PLS | 偏最小二乘法 | 一般脑-行为关联 | ✅ 完整 | 固定n_components |
| Adaptive-PLS | 自适应偏最小二乘法 | 自动选择最优成分数量 | ✅ 完整 | 内部CV确定n_components |
| Sparse-CCA | 稀疏典型相关分析 | 特征选择和可解释性 | ⚠️ 回退到CCA | 稀疏正则化 |

### ⚙️ 配置

编辑 `src/models/config.json` 自定义：
- 数据路径和质量阈值
- 模型参数和默认值
- 评估设置（CV、置换）
- 输出格式和位置
- 日志配置
- HPC优化设置

### 📚 关键类和函数

#### 数据加载
- `EFNYDataLoader`: 加载脑和行为数据
- `create_synthetic_data`: 生成测试数据

#### 预处理
- `ConfoundRegressor`: Sklearn兼容的混杂回归
- `DataQualityFilter`: 质量过滤和验证

#### 模型
- `BaseBrainBehaviorModel`: 所有模型的基类
- `PLSModel`: 偏最小二乘法实现
- `SparseCCAModel`: 稀疏CCA（带回退）
- `AdaptivePLSModel`: 自适应PLS（自动选择n_components）
- `create_model`: 模型创建的工厂函数

#### 评估
- `CrossValidator`: 交叉验证框架
- `PermutationTester`: 置换检验
- `run_nested_cv_evaluation`: 嵌套CV实现

#### 工具
- `setup_logging`: 配置日志
- `save_results`/`load_results`: 结果持久化
- `ConfigManager`: 配置管理

### 🎯 未来增强

- [ ] 完成Sparse-CCA实现
- [ ] 添加更多评估指标
- [ ] 实现特征重要性分析
- [ ] 添加可视化工具
- [ ] 支持更多脑分区图谱
- [ ] 集成神经影像管道（Nipype）
- [ ] 基于Web的结果可视化
- [ ] 支持纵向数据分析

## 数据流程

1. **预处理阶段**: 原始数据 → 质量控制 → 格式标准化 → 有效数据列表
2. **功能连接分析**: fMRI数据 → 脑区时间序列 → 功能连接矩阵 → 组水平分析
3. **行为指标分析**: 任务数据 → 行为指标计算 → 相似性分析 → 可视化展示
4. **脑-行为关联**: 脑数据 + 行为数据 → 混杂回归 → PLS/Sparse-CCA → 交叉验证/置换检验

## 使用说明

### 基本使用流程
```bash
# 1. 数据预处理
python src/preprocess/get_mri_sublist.py
python src/preprocess/screen_head_motion_efny.py

# 2. 功能连接计算
python src/functional_conn/compute_fc_schaefer.py
python src/functional_conn/compute_group_avg_fc.py

# 3. 行为指标分析
python src/metric_compute/compute_efny_metrics.py
python src/metric_compute/metrics_similarity_heatmap.py

# 4. 脑-行为关联分析（基础示例）
python src/models/example_usage.py
```

### 批处理作业
```bash
# 提交批处理任务
bash src/preprocess/batch_run_xcpd.sh
bash src/functional_conn/submit_compute_fc.sh

# HPC脑-行为关联分析
sbatch src/scripts/submit_hpc_job.sh
```

### 合成数据测试
```bash
# 使用合成数据测试脑-行为关联模型
python src/scripts/run_single_task.py \
    --task_id 0 \
    --model_type pls \
    --n_components 3 \
    --use_synthetic \
    --n_subjects 100 \
    --n_brain_features 200 \
    --n_behavioral_measures 15
```

## 注意事项

1. **数据格式**: 确保输入数据格式符合脚本要求
2. **路径设置**: 检查文件路径和目录结构是否正确
3. **依赖项**: 安装所需的Python包和软件依赖
4. **参数配置**: 根据具体研究需求调整参数设置
5. **质量控制**: 定期检查中间结果和输出质量
6. **HPC使用**: 确保SLURM环境配置正确，合理设置作业资源

## 输出文件

- **预处理**: 有效被试列表、质量控制报告
- **功能连接**: 功能连接矩阵、组平均结果
- **行为指标**: 任务性能指标、相似性热图
- **脑-行为关联**: 规范相关系数、成分分数、载荷矩阵、置换检验结果（JSON和NPZ格式）

## 更新日志

- 2025-12-14: 新增脑-行为关联分析模块（PLS/Sparse-CCA），支持HPC并行化
- 2025-12-11: 修复了`metrics_similarity_heatmap.py`中Flanker任务指标的显示问题，现在正确包含所有32个行为指标