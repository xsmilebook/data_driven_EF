#!/usr/bin/env python3
"""
评估模块 - 交叉验证和评估指标计算
提供通用的 CV 框架，不依赖具体模型类型
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
        self.cv_splitter = None
    
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
        
        # 创建交叉验证分割器（支持基于聚类的分层）
        if self.stratify:
            n_clusters = min(
                max(2, self.n_splits * 2),
                max(2, n_samples // max(2, self.n_splits))
            )
            logger.info(f"Using cluster-based stratified CV with {n_clusters} clusters")
            km = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = km.fit_predict(Y_values)
            self.cv_splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            split_iterator = self.cv_splitter.split(X_values, cluster_labels)
        else:
            self.cv_splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            split_iterator = self.cv_splitter.split(X_values)
        
        # 初始化结果存储
        fold_results = []
        all_predictions = []

        def _impute_with_train_mean(train_arr: np.ndarray, test_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            train_out = train_arr.copy()
            test_out = test_arr.copy()
            means = np.nanmean(train_out, axis=0)
            means = np.where(np.isfinite(means), means, 0.0)
            train_mask = np.isnan(train_out)
            if train_mask.any():
                train_out[train_mask] = np.take(means, np.where(train_mask)[1])
            test_mask = np.isnan(test_out)
            if test_mask.any():
                test_out[test_mask] = np.take(means, np.where(test_mask)[1])
            return train_out, test_out
        
        # 交叉验证循环
        for fold_idx, (train_idx, test_idx) in enumerate(split_iterator):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # 分割数据
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            Y_train, Y_test = Y_values[train_idx], Y_values[test_idx]
            
            if confounds_values is not None:
                confounds_train, confounds_test = confounds_values[train_idx], confounds_values[test_idx]
            else:
                confounds_train, confounds_test = None, None

            if confounds_train is not None:
                confounds_train, confounds_test = _impute_with_train_mean(confounds_train, confounds_test)

            X_train, X_test = _impute_with_train_mean(X_train, X_test)
            Y_train, Y_test = _impute_with_train_mean(Y_train, Y_test)
            
            # 步骤1: 在训练集上拟合混杂变量回归
            if confounds_values is not None:
                confound_regressor_X = ConfoundRegressor(standardize=True)
                confound_regressor_Y = ConfoundRegressor(standardize=True)
                X_train_clean = confound_regressor_X.fit_transform(X_train, confounds=confounds_train)
                Y_train_clean = confound_regressor_Y.fit_transform(Y_train, confounds=confounds_train)

                # 应用到测试集
                X_test_clean = confound_regressor_X.transform(X_test, confounds=confounds_test)
                Y_test_clean = confound_regressor_Y.transform(Y_test, confounds=confounds_test)
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
        
        # 获取最大成分数量（处理AdaptivePLS的情况）
        max_n_components = max(result['n_components'] for result in fold_results)
        
        # 初始化汇总数组
        mean_canonical_corrs = np.zeros(max_n_components)
        std_canonical_corrs = np.zeros(max_n_components)
        mean_var_exp_X = np.zeros(max_n_components)
        std_var_exp_X = np.zeros(max_n_components)
        mean_var_exp_Y = np.zeros(max_n_components)
        std_var_exp_Y = np.zeros(max_n_components)
        
        # 收集各折结果（填充到最大尺寸）
        all_canonical_corrs = np.full((n_folds, max_n_components), np.nan)
        all_var_exp_X = np.full((n_folds, max_n_components), np.nan)
        all_var_exp_Y = np.full((n_folds, max_n_components), np.nan)
        
        for i, fold_result in enumerate(fold_results):
            n_comp = fold_result['n_components']
            all_canonical_corrs[i, :n_comp] = fold_result['canonical_correlations'][:n_comp]
            all_var_exp_X[i, :n_comp] = fold_result['variance_explained_X'][:n_comp]
            all_var_exp_Y[i, :n_comp] = fold_result['variance_explained_Y'][:n_comp]
        
        # 计算均值和标准差（忽略NaN值）
        with np.errstate(all='ignore'):
            mean_canonical_corrs = np.nanmean(all_canonical_corrs, axis=0)
            std_canonical_corrs = np.nanstd(all_canonical_corrs, axis=0)
            mean_var_exp_X = np.nanmean(all_var_exp_X, axis=0)
            std_var_exp_X = np.nanstd(all_var_exp_X, axis=0)
            mean_var_exp_Y = np.nanmean(all_var_exp_Y, axis=0)
            std_var_exp_Y = np.nanstd(all_var_exp_Y, axis=0)
        
        # 创建汇总结果
        cv_results = {
            'n_folds': n_folds,
            'max_n_components': max_n_components,
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
        n_components = cv_results.get('max_n_components')
        if n_components is None:
            mean_corrs = cv_results.get('mean_canonical_correlations')
            n_components = len(mean_corrs) if mean_corrs is not None else 0
        
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
                           permutation_seed: Optional[int] = None,
                           cv_n_splits: int = 5,
                           cv_shuffle: bool = True,
                           cv_random_state: Optional[int] = None,
                           return_cv_results: bool = False) -> Dict[str, Any]:
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
        
        cv_seed = permutation_seed if cv_random_state is None else cv_random_state
        cv_evaluator = CrossValidator(
            n_splits=cv_n_splits,
            shuffle=cv_shuffle,
            random_state=cv_seed,
        )

        cv_results = cv_evaluator.run_cv_evaluation(
            model,
            X_values,
            Y_permuted,
            confounds=confounds,
        )

        mean_corrs = np.asarray(cv_results.get('mean_canonical_correlations', []), dtype=float)
        vector3_stat = float(np.linalg.norm(mean_corrs)) if mean_corrs.size else float('nan')

        result = {
            'permutation_seed': permutation_seed,
            'canonical_correlations': mean_corrs,
            'mean_canonical_correlations': mean_corrs,
            'cv_statistic_vector3': vector3_stat,
            'n_components': int(mean_corrs.size),
            'n_samples': int(len(X_values)),
            'cv_n_splits': int(cv_n_splits),
        }

        if return_cv_results:
            result['cv_results'] = cv_results

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

    def calculate_p_value_scalar(self, observed_statistic: float,
                                 permuted_statistics: np.ndarray) -> float:
        permuted_values = np.asarray(permuted_statistics, dtype=float)
        if permuted_values.ndim != 1:
            permuted_values = permuted_values.reshape(-1)
        n_greater = np.sum(np.abs(permuted_values) >= np.abs(observed_statistic))
        return float((n_greater + 1) / (len(permuted_values) + 1))


def run_nested_cv_evaluation(model: BaseBrainBehaviorModel,
                           X: Union[np.ndarray, pd.DataFrame],
                           Y: Union[np.ndarray, pd.DataFrame],
                           confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           outer_cv_splits: int = 5,
                           inner_cv_splits: int = 3,
                           param_grid: Optional[Union[Dict[str, list], list]] = None,
                           outer_shuffle: bool = True,
                           inner_shuffle: bool = True,
                           outer_random_state: Optional[int] = None,
                           inner_random_state: Optional[int] = None,
                           standardize_domains: bool = False,
                           pca_components_X: Optional[int] = None,
                           pca_components_Y: Optional[int] = None) -> Dict[str, Any]:
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

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    if isinstance(Y, pd.DataFrame):
        Y_values = Y.values
    else:
        Y_values = Y

    if isinstance(confounds, pd.DataFrame):
        confounds_values = confounds.values
    else:
        confounds_values = confounds

    if param_grid is None:
        param_candidates = [model.get_params()]
    else:
        if isinstance(param_grid, dict):
            param_candidates = list(ParameterGrid(param_grid))
        elif isinstance(param_grid, list):
            param_candidates = param_grid
        else:
            raise ValueError("param_grid must be a dict (ParameterGrid style), a list of dicts, or None")

    def _impute_with_train_mean(train_arr: np.ndarray, test_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        train_out = train_arr.copy()
        test_out = test_arr.copy()
        means = np.nanmean(train_out, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        train_mask = np.isnan(train_out)
        if train_mask.any():
            train_out[train_mask] = np.take(means, np.where(train_mask)[1])
        test_mask = np.isnan(test_out)
        if test_mask.any():
            test_out[test_mask] = np.take(means, np.where(test_mask)[1])
        return train_out, test_out

    def _prep_two_domain(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_apply: np.ndarray,
        Y_apply: np.ndarray,
        C_train: Optional[np.ndarray],
        C_apply: Optional[np.ndarray],
        rng_seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if C_train is not None:
            C_train, C_apply = _impute_with_train_mean(C_train, C_apply)

        X_train, X_apply = _impute_with_train_mean(X_train, X_apply)
        Y_train, Y_apply = _impute_with_train_mean(Y_train, Y_apply)

        if C_train is not None:
            reg_X = ConfoundRegressor(standardize=True)
            reg_Y = ConfoundRegressor(standardize=True)
            X_train = reg_X.fit_transform(X_train, confounds=C_train)
            Y_train = reg_Y.fit_transform(Y_train, confounds=C_train)
            X_apply = reg_X.transform(X_apply, confounds=C_apply)
            Y_apply = reg_Y.transform(Y_apply, confounds=C_apply)

        if standardize_domains:
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            Y_train = scaler_Y.fit_transform(Y_train)
            X_apply = scaler_X.transform(X_apply)
            Y_apply = scaler_Y.transform(Y_apply)

        if pca_components_X is not None:
            pca_X = PCA(n_components=pca_components_X, random_state=rng_seed)
            X_train = pca_X.fit_transform(X_train)
            X_apply = pca_X.transform(X_apply)

        if pca_components_Y is not None:
            pca_Y = PCA(n_components=pca_components_Y, random_state=rng_seed)
            Y_train = pca_Y.fit_transform(Y_train)
            Y_apply = pca_Y.transform(Y_apply)

        return X_train, Y_train, X_apply, Y_apply

    def _fit_and_score(
        params: Dict[str, Any],
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        model_fold = model.__class__(**params)
        model_fold.fit(X_train, Y_train)
        X_scores, Y_scores = model_fold.transform(X_test, Y_test)
        corrs = np.asarray(model_fold.calculate_canonical_correlations(X_scores, Y_scores), dtype=float)
        fold_score = float(corrs[0]) if corrs.size else float('nan')
        return fold_score, corrs

    outer_kf = KFold(
        n_splits=outer_cv_splits,
        shuffle=outer_shuffle,
        random_state=outer_random_state,
    )

    outer_fold_results = []
    for outer_fold_idx, (train_out_idx, test_out_idx) in enumerate(outer_kf.split(X_values)):
        X_train_out = X_values[train_out_idx]
        Y_train_out = Y_values[train_out_idx]
        X_test_out = X_values[test_out_idx]
        Y_test_out = Y_values[test_out_idx]

        if confounds_values is not None:
            C_train_out = confounds_values[train_out_idx]
            C_test_out = confounds_values[test_out_idx]
        else:
            C_train_out, C_test_out = None, None

        inner_kf = KFold(
            n_splits=inner_cv_splits,
            shuffle=inner_shuffle,
            random_state=inner_random_state,
        )

        theta_records = []
        best_theta = param_candidates[0]
        best_mean_corr = -np.inf

        for theta in param_candidates:
            inner_fold_scores = []
            inner_fold_corrs = []

            for inner_fold_idx, (train_in_rel, val_in_rel) in enumerate(inner_kf.split(X_train_out)):
                train_in_idx = train_out_idx[train_in_rel]
                val_in_idx = train_out_idx[val_in_rel]

                X_train_in = X_values[train_in_idx]
                Y_train_in = Y_values[train_in_idx]
                X_val_in = X_values[val_in_idx]
                Y_val_in = Y_values[val_in_idx]

                if confounds_values is not None:
                    C_train_in = confounds_values[train_in_idx]
                    C_val_in = confounds_values[val_in_idx]
                else:
                    C_train_in, C_val_in = None, None

                rng_seed = None
                if inner_random_state is not None:
                    rng_seed = int(inner_random_state + outer_fold_idx * 1000 + inner_fold_idx)

                X_train_in_p, Y_train_in_p, X_val_in_p, Y_val_in_p = _prep_two_domain(
                    X_train_in,
                    Y_train_in,
                    X_val_in,
                    Y_val_in,
                    C_train_in,
                    C_val_in,
                    rng_seed,
                )

                fold_score, fold_corrs = _fit_and_score(theta, X_train_in_p, Y_train_in_p, X_val_in_p, Y_val_in_p)
                inner_fold_scores.append(fold_score)
                inner_fold_corrs.append(fold_corrs)

            mean_corr = float(np.nanmean(np.asarray(inner_fold_scores, dtype=float)))
            std_corr = float(np.nanstd(np.asarray(inner_fold_scores, dtype=float)))
            theta_records.append(
                {
                    'params': theta,
                    'mean_corr': mean_corr,
                    'std_corr': std_corr,
                    'inner_fold_scores': inner_fold_scores,
                }
            )

            if np.isfinite(mean_corr) and mean_corr > best_mean_corr:
                best_mean_corr = mean_corr
                best_theta = theta

        rng_seed_outer = outer_random_state
        if rng_seed_outer is not None:
            rng_seed_outer = int(rng_seed_outer + outer_fold_idx)

        X_train_out_p, Y_train_out_p, X_test_out_p, Y_test_out_p = _prep_two_domain(
            X_train_out,
            Y_train_out,
            X_test_out,
            Y_test_out,
            C_train_out,
            C_test_out,
            rng_seed_outer,
        )

        test_score, test_corrs = _fit_and_score(best_theta, X_train_out_p, Y_train_out_p, X_test_out_p, Y_test_out_p)
        outer_fold_results.append(
            {
                'outer_fold': outer_fold_idx,
                'train_out_idx': train_out_idx,
                'test_out_idx': test_out_idx,
                'best_params': best_theta,
                'best_inner_mean_corr': best_mean_corr,
                'inner_cv_table': theta_records,
                'test_mean_corr': test_score,
                'test_canonical_correlations': test_corrs,
            }
        )

    outer_scores = np.asarray([r.get('test_mean_corr', np.nan) for r in outer_fold_results], dtype=float)

    max_n_components = 0
    for r in outer_fold_results:
        corrs = r.get('test_canonical_correlations')
        if corrs is None:
            continue
        try:
            max_n_components = max(max_n_components, int(np.asarray(corrs).size))
        except Exception:
            continue

    all_test_corrs = np.full((len(outer_fold_results), max_n_components), np.nan, dtype=float)
    for i, r in enumerate(outer_fold_results):
        corrs = r.get('test_canonical_correlations')
        if corrs is None:
            continue
        arr = np.asarray(corrs, dtype=float).reshape(-1)
        n = min(arr.size, max_n_components)
        if n > 0:
            all_test_corrs[i, :n] = arr[:n]

    with np.errstate(all='ignore'):
        outer_mean_canonical_correlations = np.nanmean(all_test_corrs, axis=0) if max_n_components else np.array([])
        outer_std_canonical_correlations = np.nanstd(all_test_corrs, axis=0) if max_n_components else np.array([])

    nested_results = {
        'n_outer_splits': int(outer_cv_splits),
        'n_inner_splits': int(inner_cv_splits),
        'outer_fold_results': outer_fold_results,
        'outer_mean_score': float(np.nanmean(outer_scores)) if outer_scores.size else float('nan'),
        'outer_std_score': float(np.nanstd(outer_scores)) if outer_scores.size else float('nan'),
        'outer_mean_canonical_correlations': outer_mean_canonical_correlations,
        'outer_std_canonical_correlations': outer_std_canonical_correlations,
        'outer_all_test_canonical_correlations': all_test_corrs,
        'nested_evaluation_completed': True,
        'param_grid_size': int(len(param_candidates)),
        'standardize_domains': bool(standardize_domains),
        'pca_components_X': pca_components_X,
        'pca_components_Y': pca_components_Y,
    }

    return nested_results
