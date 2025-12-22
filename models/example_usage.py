#!/usr/bin/env python3
"""
使用示例 - 展示如何使用重构后的模块化 PLS 分析管道
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models import (
    EFNYDataLoader, create_synthetic_data, ConfoundRegressor,
    create_model, CrossValidator, PermutationTester,
    setup_logging, save_results, get_available_models
)


def example_basic_analysis():
    """基本分析示例"""
    print("\n" + "="*60)
    print("示例 1: 基本 PLS 分析")
    print("="*60)
    
    # 创建合成数据
    brain_data, behavioral_data, covariates = create_synthetic_data(
        n_subjects=100, n_brain_features=100, n_behavioral_measures=20
    )
    
    print(f"数据形状 - 脑: {brain_data.shape}, 行为: {behavioral_data.shape}")
    
    # 创建调参版本 PLS 模型（此处固定 n_components_range=[5]）
    pls_model = create_model('adaptive_pls', n_components_range=[5], random_state=42)
    
    # 拟合模型
    pls_model.fit(brain_data, behavioral_data)
    
    # 获取结果
    X_scores, Y_scores = pls_model.transform(brain_data, behavioral_data)
    canonical_corrs = pls_model.calculate_canonical_correlations(X_scores, Y_scores)
    variance_explained = pls_model.calculate_variance_explained(
        brain_data, behavioral_data, X_scores, Y_scores
    )
    
    # 显示结果
    print("\n典型相关系数:")
    for i, corr in enumerate(canonical_corrs):
        print(f"  成分 {i+1}: {corr:.4f}")
    
    print("\n行为方差解释:")
    for i, var_exp in enumerate(variance_explained['variance_explained_Y']):
        print(f"  成分 {i+1}: {var_exp:.2f}%")


def example_cross_validation():
    """交叉验证示例"""
    print("\n" + "="*60)
    print("示例 2: 交叉验证")
    print("="*60)
    
    # 创建数据
    brain_data, behavioral_data, covariates = create_synthetic_data(
        n_subjects=200, n_brain_features=200, n_behavioral_measures=15
    )
    
    # 创建模型
    model = create_model('adaptive_pls', n_components_range=[5], random_state=42)
    
    # 创建交叉验证器
    cv = CrossValidator(n_splits=5, shuffle=True, random_state=42)
    
    # 运行交叉验证
    cv_results = cv.run_cv_evaluation(model, brain_data, behavioral_data, confounds=covariates)
    
    # 创建汇总表
    summary_df = cv.create_cv_summary_table(cv_results)
    
    print("\n交叉验证结果:")
    print(summary_df.to_string(index=False))


def example_permutation_test():
    """置换检验示例"""
    print("\n" + "="*60)
    print("示例 3: 置换检验")
    print("="*60)
    
    # 创建数据
    brain_data, behavioral_data, covariates = create_synthetic_data(
        n_subjects=150, n_brain_features=150, n_behavioral_measures=10
    )
    
    # 创建模型
    model = create_model('adaptive_pls', n_components_range=[5], random_state=42)
    
    # 运行真实数据分析
    model.fit(brain_data, behavioral_data)
    X_scores, Y_scores = model.transform(brain_data, behavioral_data)
    real_correlations = model.calculate_canonical_correlations(X_scores, Y_scores)
    
    # 运行置换检验
    perm_tester = PermutationTester(n_permutations=100, random_state=42)
    
    perm_correlations = []
    for i in range(100):
        perm_result = perm_tester.run_permutation_test(
            model, brain_data, behavioral_data, confounds=covariates, permutation_seed=42+i
        )
        perm_correlations.append(perm_result['canonical_correlations'])
    
    perm_correlations = np.array(perm_correlations)
    
    # 计算 p 值
    p_values = perm_tester.calculate_p_values(real_correlations, perm_correlations)
    
    print("\n置换检验结果:")
    print("成分 | 真实相关 | p 值")
    print("-" * 25)
    for i, (real_corr, p_val) in enumerate(zip(real_correlations, p_values)):
        print(f"  {i+1}  |   {real_corr:.4f}   | {p_val:.4f}")


def example_available_models():
    """可用模型展示"""
    print("\n" + "="*60)
    print("示例 4: 可用模型")
    print("="*60)
    
    available_models = get_available_models()
    print(f"支持的模型类型: {available_models}")
    
    for model_type in available_models:
        print(f"\n{model_type.upper()} 模型参数:")
        model = create_model(model_type)
        model_info = model.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")


def example_real_data_loading():
    """真实数据加载示例"""
    print("\n" + "="*60)
    print("示例 5: 真实数据加载（需要实际数据文件）")
    print("="*60)
    
    try:
        # 创建数据加载器
        data_loader = EFNYDataLoader()
        
        # 获取可用的行为指标
        available_measures = data_loader.get_available_behavioral_measures()
        print(f"可用的行为指标数量: {len(available_measures)}")
        print("前10个行为指标:")
        for i, measure in enumerate(available_measures[:10]):
            print(f"  {i+1}. {measure}")
        
        # 尝试加载所有数据
        brain_data, behavioral_data, subject_ids = data_loader.load_all_data()
        
        print(f"\n数据加载成功:")
        print(f"  被试数量: {len(subject_ids)}")
        print(f"  脑数据形状: {brain_data.shape}")
        print(f"  行为数据形状: {behavioral_data.shape}")
        
    except FileNotFoundError as e:
        print(f"真实数据文件未找到: {e}")
        print("请确保数据文件在正确的位置，或使用合成数据进行测试")


def example_save_and_load_results():
    """结果保存和加载示例"""
    print("\n" + "="*60)
    print("示例 6: 结果保存和加载")
    print("="*60)
    
    # 创建示例结果
    results = {
        'model_type': 'pls',
        'n_components': 3,
        'canonical_correlations': np.array([0.65, 0.42, 0.28]),
        'variance_explained_X': np.array([8.5, 12.3, 15.1]),
        'variance_explained_Y': np.array([22.1, 35.6, 42.8]),
        'timestamp': '20241214_143000',
        'metadata': {
            'n_samples': 100,
            'n_features_X': 200,
            'n_features_Y': 15
        }
    }
    
    # 保存结果
    # 保存结果
    from models.utils import save_results, load_results
    
    output_path = Path("example_results")
    saved_files = save_results(results, output_path)
    
    print("保存的文件:")
    for format_type, file_path in saved_files.items():
        print(f"  {format_type}: {file_path}")
    
    # 加载结果
    if 'json' in saved_files:
        loaded_results = load_results(saved_files['json'])
        print(f"\n从 JSON 加载的结果:")
        print(f"  模型类型: {loaded_results['model_type']}")
        print(f"  成分数量: {loaded_results['n_components']}")
        print(f"  典型相关: {loaded_results['canonical_correlations']}")


def example_adaptive_pls():
    """自适应PLS模型示例 - 自动选择n_components"""
    print("\n" + "="*60)
    print("示例 7: 自适应PLS模型（自动选择n_components）")
    print("="*60)
    
    # 创建数据
    brain_data, behavioral_data, covariates = create_synthetic_data(
        n_subjects=200, n_brain_features=150, n_behavioral_measures=25
    )
    
    # 预处理
    confound_regressor = ConfoundRegressor(standardize=True)
    brain_clean = confound_regressor.fit_transform(brain_data, confounds=covariates)
    behavioral_clean = confound_regressor.fit_transform(behavioral_data, confounds=covariates)
    
    # 创建自适应PLS模型
    adaptive_model = create_model(
        'adaptive_pls', 
        n_components_range=[1, 2, 3, 4, 5, 6],  # 搜索范围
        cv_folds=5,                            # 内部CV折数
        criterion='canonical_correlation',     # 选择标准
        random_state=42
    )
    
    print(f"开始自适应成分选择，搜索范围: {adaptive_model.n_components_range}")
    
    # 拟合模型（会自动选择最优n_components）
    adaptive_model.fit(brain_clean, behavioral_clean)
    
    # 获取结果
    X_scores, Y_scores = adaptive_model.transform(brain_clean, behavioral_clean)
    canonical_corrs = adaptive_model.calculate_canonical_correlations(X_scores, Y_scores)
    variance_explained = adaptive_model.calculate_variance_explained(
        brain_clean, behavioral_clean, X_scores, Y_scores
    )
    
    # 显示自适应选择结果
    model_info = adaptive_model.get_model_info()
    cv_results = adaptive_model.get_cv_results()
    
    print(f"\n自适应选择结果:")
    print(f"  最优成分数量: {model_info['optimal_n_components']}")
    print(f"  选择标准: {model_info['criterion']}")
    print(f"  内部CV折数: {model_info['cv_folds']}")
    
    print(f"\n各成分数量评估结果:")
    print("n_components | 典型相关 | 方差解释X% | 方差解释Y%")
    print("-" * 50)
    for n_comp, metrics in cv_results.items():
        print(f"     {n_comp}      |   {metrics['canonical_correlation']:.4f}   |    {metrics['variance_explained_X']:.2f}     |    {metrics['variance_explained_Y']:.2f}")
    
    print(f"\n最终模型结果（n_components={model_info['optimal_n_components']}）:")
    print("典型相关系数:")
    for i, corr in enumerate(canonical_corrs):
        print(f"  成分 {i+1}: {corr:.4f}")
    
    print("\n行为方差解释:")
    for i, var_exp in enumerate(variance_explained['variance_explained_Y']):
        print(f"  成分 {i+1}: {var_exp:.2f}%")


def main():
    """运行所有示例"""
    print("EFNY Brain-Behavior Analysis - 使用示例")
    print("="*60)
    
    # 设置日志
    setup_logging(log_level="INFO")
    
    # 运行示例
    try:
        example_basic_analysis()
        example_cross_validation()
        example_permutation_test()
        example_available_models()
        example_real_data_loading()
        example_save_and_load_results()
        example_adaptive_pls()  # 新增的自适应PLS示例
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 导入 numpy（用于示例）
    import numpy as np
    
    main()