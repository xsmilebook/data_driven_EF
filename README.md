# 数据驱动执行功能研究代码库文档

本代码库包含用于数据驱动执行功能（Executive Function, EF）研究的分析工具和脚本。主要分为三个功能模块：预处理、功能连接分析和指标计算。

## 目录结构

```
src/
├── preprocess/          # 数据预处理模块
├── functional_conn/     # 功能连接分析模块
└── metric_compute/      # 行为指标计算模块
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

## 数据流程

1. **预处理阶段**: 原始数据 → 质量控制 → 格式标准化 → 有效数据列表
2. **功能连接分析**: fMRI数据 → 脑区时间序列 → 功能连接矩阵 → 组水平分析
3. **行为指标分析**: 任务数据 → 行为指标计算 → 相似性分析 → 可视化展示

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
```

### 批处理作业
```bash
# 提交批处理任务
bash src/preprocess/batch_run_xcpd.sh
bash src/functional_conn/submit_compute_fc.sh
```

## 注意事项

1. **数据格式**: 确保输入数据格式符合脚本要求
2. **路径设置**: 检查文件路径和目录结构是否正确
3. **依赖项**: 安装所需的Python包和软件依赖
4. **参数配置**: 根据具体研究需求调整参数设置
5. **质量控制**: 定期检查中间结果和输出质量

## 输出文件

- **预处理**: 有效被试列表、质量控制报告
- **功能连接**: 功能连接矩阵、组平均结果
- **行为指标**: 任务性能指标、相似性热图

## 更新日志

- 2025-12-11: 修复了`metrics_similarity_heatmap.py`中Flanker任务指标的显示问题，现在正确包含所有32个行为指标