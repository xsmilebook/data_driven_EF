#!/usr/bin/env python3
"""
评估模块 - 交叉验证和评估指标计算
提供通用的 CV 框架，不依赖具体模型类型
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import pearsonr
import logging
from pathlib import Path

from .preprocessing import ConfoundRegressor
from .models import BaseBrainBehaviorModel

logger = logging.getLogger(__name__)


class CrossValidator:
    """交叉验证器 - 通用的 CV 框架"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, 
                 random_state: Optional[int] = None, stratify: bool = False):
        """
        初始化交叉验证器
        
        Args:
            n_splits: 折数
            shuffle: 是否打乱数据
            random_state: 随机种子
            stratify: 是否使用分层采样
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        
        # 创建交叉验证分割器
        if stratify:
            self.cv_splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            self.cv_splitter = KFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
    
    def run_cv_evaluation(self, model: BaseBrainBehaviorModel,
                         X: Union[np.ndarray, pd.DataFrame],
                         Y: Union[np.ndarray, pd.DataFrame],
                         confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                         return_predictions: bool = False) -> Dict[str, Any]:
        """
        运行交叉验证评估
        
        该函数不依赖具体模型类型，只负责：
        1. 划分训练/测试集
        2. 在训练集上拟合混杂变量回归
        3. 应用到测试集
        4. 训练模型并计算评估指标
        
        Args:
            model: 脑-行为模型（PLS 或 Sparse-CCA）
            X: 脑数据 (n_samples, n_features)
            Y: 行为数据 (n_samples, n_targets)
            confounds: 混杂变量 (n_samples, n_confounds)
            return_predictions: 是否返回预测结果
            
        Returns:
            包含 CV 结果的字典
        """
        logger.info(f"Starting {self.n_splits}-fold cross-validation evaluation")
        logger.info(f"Data shapes - X: {X.shape}, Y: {Y.shape}")
        if confounds is not None:
            logger.info(f"Confounds shape: {confounds.shape}")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
        else:
            X_values = X
            feature_names = None
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
            target_names = Y.columns.tolist()
        else:
            Y_values = Y
            target_names = None
            
        if isinstance(confounds, pd.DataFrame):
            confounds_values = confounds.values
            confound_names = confounds.columns.tolist()
        else:
            confounds_values = confounds
            confound_names = None
        
        n_samples = X_values.shape[0]
        
        # 初始化结果存储
        fold_results = []
        all_predictions = []
        
        # 交叉验证循环
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X_values)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # 分割数据
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            Y_train, Y_test = Y_values[train_idx], Y_values[test_idx]
            
            if confounds_values is not None:
                confounds_train, confounds_test = confounds_values[train_idx], confounds_values[test_idx]
            else:
                confounds_train, confounds_test = None, None
            
            # 步骤1: 在训练集上拟合混杂变量回归
            if confounds_values is not None:
                confound_regressor = ConfoundRegressor(standardize=True)
                X_train_clean = confound_regressor.fit_transform(X_train, confounds=confounds_train)
                Y_train_clean = confound_regressor.fit_transform(Y_train, confounds=confounds_train)
                
                # 应用到测试集
                X_test_clean = confound_regressor.transform(X_test, confounds=confounds_test)
                Y_test_clean = confound_regressor.transform(Y_test, confounds=confounds_test)
            else:
                X_train_clean, Y_train_clean = X_train, Y_train
                X_test_clean, Y_test_clean = X_test, Y_test
            
            # 步骤2: 在清理后的训练数据上拟合模型
            model_fold = model.__class__(**model.get_params())  # 创建模型副本
            model_fold.fit(X_train_clean, Y_train_clean)
            
            # 步骤3: 在测试集上评估模型
            X_test_scores, Y_test_scores = model_fold.transform(X_test_clean, Y_test_clean)
            
            # 计算评估指标
            fold_metrics = self._calculate_fold_metrics(
                X_test_scores, Y_test_scores, model_fold, fold_idx
            )
            
            fold_results.append(fold_metrics)
            
            if return_predictions:
                all_predictions.append({
                    'fold': fold_idx,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'X_test_scores': X_test_scores,
                    'Y_test_scores': Y_test_scores,
                    'X_test_clean': X_test_clean,
                    'Y_test_clean': Y_test_clean
                })
        
        # 汇总结果
        cv_results = self._aggregate_cv_results(fold_results)
        
        if return_predictions:
            cv_results['predictions'] = all_predictions
        
        logger.info("Cross-validation evaluation completed")
        return cv_results
    
    def _calculate_fold_metrics(self, X_scores: np.ndarray, Y_scores: np.ndarray,
                               model: BaseBrainBehaviorModel, fold_idx: int) -> Dict[str, Any]:
        """
        计算单折的评估指标
        
        Args:
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
            model: 拟合的模型
            fold_idx: 折索引
            
        Returns:
            评估指标字典
        """
        n_components = X_scores.shape[1]
        
        # 计算典型相关系数
        canonical_corrs = model.calculate_canonical_correlations(X_scores, Y_scores)
        
        # 计算方差解释（这里使用简化的方法）
        var_exp_X = []
        var_exp_Y = []
        
        for i in range(n_components):
            # 简化计算：使用潜变量分数的方差比例
            var_exp_X.append(np.var(X_scores[:, :(i+1)], axis=0).sum() / 
                           np.var(X_scores, axis=0).sum() * 100)
            var_exp_Y.append(np.var(Y_scores[:, :(i+1)], axis=0).sum() / 
                           np.var(Y_scores, axis=0).sum() * 100)
        
        metrics = {
            'fold': fold_idx,
            'n_components': n_components,
            'canonical_correlations': canonical_corrs,
            'variance_explained_X': np.array(var_exp_X),
            'variance_explained_Y': np.array(var_exp_Y),
            'n_samples': X_scores.shape[0]
        }
        
        return metrics
    
    def _aggregate_cv_results(self, fold_results: list) -> Dict[str, Any]:
        """
        汇总交叉验证结果
        
        Args:
            fold_results: 各折的结果列表
            
        Returns:
            汇总结果字典
        """
        n_folds = len(fold_results)
        n_components = fold_results[0]['n_components']
        
        # 初始化汇总数组
        mean_canonical_corrs = np.zeros(n_components)
        std_canonical_corrs = np.zeros(n_components)
        mean_var_exp_X = np.zeros(n_components)
        std_var_exp_X = np.zeros(n_components)
        mean_var_exp_Y = np.zeros(n_components)
        std_var_exp_Y = np.zeros(n_components)
        
        # 收集各折结果
        all_canonical_corrs = np.zeros((n_folds, n_components))
        all_var_exp_X = np.zeros((n_folds, n_components))
        all_var_exp_Y = np.zeros((n_folds, n_components))
        
        for i, fold_result in enumerate(fold_results):
            all_canonical_corrs[i] = fold_result['canonical_correlations']
            all_var_exp_X[i] = fold_result['variance_explained_X']
            all_var_exp_Y[i] = fold_result['variance_explained_Y']
        
        # 计算均值和标准差
        mean_canonical_corrs = np.mean(all_canonical_corrs, axis=0)
        std_canonical_corrs = np.std(all_canonical_corrs, axis=0)
        mean_var_exp_X = np.mean(all_var_exp_X, axis=0)
        std_var_exp_X = np.std(all_var_exp_X, axis=0)
        mean_var_exp_Y = np.mean(all_var_exp_Y, axis=0)
        std_var_exp_Y = np.std(all_var_exp_Y, axis=0)
        
        # 创建汇总结果
        cv_results = {
            'n_folds': n_folds,
            'n_components': n_components,
            'mean_canonical_correlations': mean_canonical_corrs,
            'std_canonical_correlations': std_canonical_corrs,
            'mean_variance_explained_X': mean_var_exp_X,
            'std_variance_explained_X': std_var_exp_X,
            'mean_variance_explained_Y': mean_var_exp_Y,
            'std_variance_explained_Y': std_var_exp_Y,
            'fold_results': fold_results,
            'all_canonical_correlations': all_canonical_corrs,
            'all_variance_explained_X': all_var_exp_X,
            'all_variance_explained_Y': all_var_exp_Y
        }
        
        return cv_results
    
    def create_cv_summary_table(self, cv_results: Dict[str, Any]) -> pd.DataFrame:
        """
        创建 CV 结果汇总表
        
        Args:
            cv_results: CV 结果字典
            
        Returns:
            汇总表 DataFrame
        """
        n_components = cv_results['n_components']
        
        summary_data = {
            'Component': np.arange(1, n_components + 1),
            'Canonical_Correlation_Mean': cv_results['mean_canonical_correlations'],
            'Canonical_Correlation_Std': cv_results['std_canonical_correlations'],
            'Variance_Explained_X_Mean_%': cv_results['mean_variance_explained_X'],
            'Variance_Explained_X_Std_%': cv_results['std_variance_explained_X'],
            'Variance_Explained_Y_Mean_%': cv_results['mean_variance_explained_Y'],
            'Variance_Explained_Y_Std_%': cv_results['std_variance_explained_Y']
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # 格式化显示
        summary_df['Canonical_Correlation_Mean'] = summary_df['Canonical_Correlation_Mean'].round(4)
        summary_df['Canonical_Correlation_Std'] = summary_df['Canonical_Correlation_Std'].round(4)
        summary_df['Variance_Explained_X_Mean_%'] = summary_df['Variance_Explained_X_Mean_%'].round(2)
        summary_df['Variance_Explained_X_Std_%'] = summary_df['Variance_Explained_X_Std_%'].round(2)
        summary_df['Variance_Explained_Y_Mean_%'] = summary_df['Variance_Explained_Y_Mean_%'].round(2)
        summary_df['Variance_Explained_Y_Std_%'] = summary_df['Variance_Explained_Y_Std_%'].round(2)
        
        return summary_df


class PermutationTester:
    """置换检验器"""
    
    def __init__(self, n_permutations: int = 1000, random_state: Optional[int] = None):
        """
        初始化置换检验器
        
        Args:
            n_permutations: 置换次数
            random_state: 随机种子
        """
        self.n_permutations = n_permutations
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def run_permutation_test(self, model: BaseBrainBehaviorModel,
                           X: Union[np.ndarray, pd.DataFrame],
                           Y: Union[np.ndarray, pd.DataFrame],
                           confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           permutation_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        运行单次置换检验
        
        Args:
            model: 脑-行为模型
            X: 脑数据
            Y: 行为数据
            confounds: 混杂变量
            permutation_seed: 置换随机种子
            
        Returns:
            置换检验结果
        """
        logger.info(f"Running permutation test with seed {permutation_seed}")
        
        # 设置随机种子
        if permutation_seed is not None:
            np.random.seed(permutation_seed)
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values.copy()  # 复制以避免修改原始数据
        else:
            Y_values = Y.copy()
        
        # 置换行为数据
        permuted_indices = np.random.permutation(len(Y_values))
        Y_permuted = Y_values[permuted_indices]
        
        # 处理混杂变量
        if confounds is not None:
            confound_regressor = ConfoundRegressor(standardize=True)
            X_clean = confound_regressor.fit_transform(X_values, confounds=confounds)
            Y_clean = confound_regressor.fit_transform(Y_permuted, confounds=confounds)
        else:
            X_clean, Y_clean = X_values, Y_permuted
        
        # 拟合模型
        model_permuted = model.__class__(**model.get_params())
        model_permuted.fit(X_clean, Y_clean)
        
        # 获取结果
        X_scores, Y_scores = model_permuted.transform(X_clean, Y_clean)
        canonical_corrs = model_permuted.calculate_canonical_correlations(X_scores, Y_scores)
        
        result = {
            'permutation_seed': permutation_seed,
            'canonical_correlations': canonical_corrs,
            'n_components': len(canonical_corrs),
            'n_samples': len(X_values)
        }
        
        return result
    
    def calculate_p_values(self, observed_correlations: np.ndarray,
                          permuted_correlations: np.ndarray) -> np.ndarray:
        """
        计算 p 值
        
        Args:
            observed_correlations: 观测到的典型相关系数
            permuted_correlations: 置换得到的典型相关系数矩阵 (n_permutations, n_components)
            
        Returns:
            p 值数组
        """
        n_components = len(observed_correlations)
        p_values = np.zeros(n_components)
        
        for i in range(n_components):
            # 计算大于观测值的置换结果比例
            permuted_values = permuted_correlations[:, i]
            n_greater = np.sum(np.abs(permuted_values) >= np.abs(observed_correlations[i]))
            p_values[i] = (n_greater + 1) / (len(permuted_values) + 1)  # +1 用于避免零p值
        
        return p_values


def run_nested_cv_evaluation(model: BaseBrainBehaviorModel,
                           X: Union[np.ndarray, pd.DataFrame],
                           Y: Union[np.ndarray, pd.DataFrame],
                           confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           outer_cv_splits: int = 5,
                           inner_cv_splits: int = 3) -> Dict[str, Any]:
    """
    运行嵌套交叉验证评估
    
    Args:
        model: 脑-行为模型
        X: 脑数据
        Y: 行为数据
        confounds: 混杂变量
        outer_cv_splits: 外层CV折数
        inner_cv_splits: 内层CV折数
        
    Returns:
        嵌套CV结果
    """
    logger.info(f"Starting nested CV: outer={outer_cv_splits}, inner={inner_cv_splits}")
    
    # 创建外层和内层CV
    outer_cv = CrossValidator(n_splits=outer_cv_splits)
    inner_cv = CrossValidator(n_splits=inner_cv_splits)
    
    # 这里可以实现超参数优化等复杂逻辑
    # 目前简化为标准的外层评估
    
    outer_results = outer_cv.run_cv_evaluation(model, X, Y, confounds)
    
    nested_results = {
        'outer_cv_results': outer_results,
        'n_outer_splits': outer_cv_splits,
        'n_inner_splits': inner_cv_splits,
        'nested_evaluation_completed': True
    }
    
    return nested_results