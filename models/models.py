#!/usr/bin/env python3
"""
模型接口模块 - 统一 PLS 和 Sparse-CCA 的接口
提供一致的模型调用方式
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class BaseBrainBehaviorModel(ABC):
    """
    脑-行为关联模型的基类
    定义统一的接口，支持 PLS 和 Sparse-CCA
    """
    
    def __init__(self, n_components: int = 5, random_state: Optional[int] = None):
        """
        初始化基类
        
        Args:
            n_components: 成分数量
            random_state: 随机种子
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame]) -> 'BaseBrainBehaviorModel':
        """
        拟合模型
        
        Args:
            X: 脑数据 (n_samples, n_brain_features)
            Y: 行为数据 (n_samples, n_behavioral_features)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  Y: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据到潜变量空间
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
        """
        pass
    
    @abstractmethod
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings: 脑载荷矩阵
            Y_loadings: 行为载荷矩阵
        """
        pass
    
    def get_scores(self, X: Union[np.ndarray, pd.DataFrame], 
                   Y: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取潜变量分数（拟合 + 转换）
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            X_scores, Y_scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting scores")
        return self.transform(X, Y)
    
    def calculate_canonical_correlations(self, X_scores: np.ndarray, 
                                         Y_scores: np.ndarray) -> np.ndarray:
        """
        计算典型相关系数
        
        Args:
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
            
        Returns:
            典型相关系数数组
        """
        correlations = []
        for i in range(self.n_components):
            corr, _ = pearsonr(X_scores[:, i], Y_scores[:, i])
            correlations.append(corr)
        return np.array(correlations)
    
    def calculate_variance_explained(self, X: Union[np.ndarray, pd.DataFrame],
                                   Y: Union[np.ndarray, pd.DataFrame],
                                   X_scores: np.ndarray,
                                   Y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算方差解释比例（使用 R² 方法）
        
        Args:
            X: 原始脑数据
            Y: 原始行为数据
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
            
        Returns:
            包含方差解释比例的字典
        """
        # 转换数据
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        # 计算总方差
        total_var_X = np.sum(np.var(X_values, axis=0))
        total_var_Y = np.sum(np.var(Y_values, axis=0))
        
        # 获取载荷
        X_loadings, Y_loadings = self.get_loadings()
        
        var_exp_X = []
        var_exp_Y = []
        
        for i in range(self.n_components):
            # 使用前 i+1 个成分重构数据
            X_reconstructed = X_scores[:, :(i+1)] @ X_loadings[:, :(i+1)].T
            Y_reconstructed = Y_scores[:, :(i+1)] @ Y_loadings[:, :(i+1)].T
            
            # 计算重构数据的方差
            var_reconstructed_X = np.sum(np.var(X_reconstructed, axis=0))
            var_reconstructed_Y = np.sum(np.var(Y_reconstructed, axis=0))
            
            # 计算解释比例
            var_exp_X.append((var_reconstructed_X / total_var_X) * 100)
            var_exp_Y.append((var_reconstructed_Y / total_var_Y) * 100)
        
        return {
            'variance_explained_X': np.array(var_exp_X),
            'variance_explained_Y': np.array(var_exp_Y)
        }


class PLSModel(BaseBrainBehaviorModel):
    """
    偏最小二乘 (PLS) 模型
    基于 sklearn.cross_decomposition.PLSCanonical
    """
    
    def __init__(self, n_components: int = 5, scale: bool = True, 
                 max_iter: int = 500, tol: float = 1e-06,
                 random_state: Optional[int] = None):
        """
        初始化 PLS 模型
        
        Args:
            n_components: 成分数量
            scale: 是否标准化数据
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        super().__init__(n_components, random_state)
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        
        # 初始化 sklearn PLS 模型
        self.model = PLSCanonical(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame]) -> 'PLSModel':
        """
        拟合 PLS 模型
        
        Args:
            X: 脑数据 (n_samples, n_brain_features)
            Y: 行为数据 (n_samples, n_behavioral_features)
            
        Returns:
            self
        """
        logger.info(f"Fitting PLS model with {self.n_components} components")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        # 拟合模型
        self.model.fit(X_values, Y_values)
        self.is_fitted = True
        
        logger.info("PLS model fitting completed")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  Y: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据到潜变量空间
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        # 转换到潜变量空间
        X_scores, Y_scores = self.model.transform(X_values, Y_values)
        
        return X_scores, Y_scores
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings: 脑载荷矩阵 (n_brain_features, n_components)
            Y_loadings: 行为载荷矩阵 (n_behavioral_features, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        return self.model.x_loadings_, self.model.y_loadings_
    
    def get_model_info(self) -> Dict[str, Union[int, bool, float]]:
        """
        获取模型信息
        
        Returns:
            模型参数字典
        """
        return {
            'model_type': 'PLS',
            'n_components': self.n_components,
            'scale': self.scale,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'is_fitted': self.is_fitted
        }


class SparseCCAModel(BaseBrainBehaviorModel):
    """
    稀疏典型相关分析 (Sparse-CCA) 模型
    预留接口，待实现
    """
    
    def __init__(self, n_components: int = 5, 
                 sparsity_X: float = 0.1, sparsity_Y: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-06,
                 random_state: Optional[int] = None):
        """
        初始化 Sparse-CCA 模型
        
        Args:
            n_components: 成分数量
            sparsity_X: 脑数据的稀疏性参数
            sparsity_Y: 行为数据的稀疏性参数
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
        """
        super().__init__(n_components, random_state)
        self.sparsity_X = sparsity_X
        self.sparsity_Y = sparsity_Y
        self.max_iter = max_iter
        self.tol = tol
        
        # TODO: 实现 Sparse-CCA 算法
        # 这里可以使用 sklearn.cross_decomposition.CCA 作为基础
        # 或者集成其他稀疏CCA实现
        logger.warning("Sparse-CCA model is not fully implemented yet")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame]) -> 'SparseCCAModel':
        """
        拟合 Sparse-CCA 模型
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            self
        """
        logger.warning("Sparse-CCA fitting not implemented yet, falling back to standard CCA")
        
        # 临时使用标准CCA作为回退
        from sklearn.cross_decomposition import CCA
        self.model = CCA(n_components=self.n_components)
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        self.model.fit(X_values, Y_values)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  Y: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            X_scores, Y_scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        return self.model.transform(X_values, Y_values)
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings, Y_loadings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        # 对于CCA，载荷矩阵就是成分向量
        return self.model.x_rotations_, self.model.y_rotations_
    
    def get_model_info(self) -> Dict[str, Union[int, float, str]]:
        """
        获取模型信息
        
        Returns:
            模型参数字典
        """
        return {
            'model_type': 'Sparse-CCA',
            'n_components': self.n_components,
            'sparsity_X': self.sparsity_X,
            'sparsity_Y': self.sparsity_Y,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'is_fitted': self.is_fitted,
            'implementation': 'Fallback to standard CCA'
        }


def create_model(model_type: str, **kwargs) -> BaseBrainBehaviorModel:
    """
    工厂函数 - 创建模型实例
    
    Args:
        model_type: 模型类型 ('pls', 'scca', 'adaptive_pls')
        **kwargs: 模型参数
        
    Returns:
        模型实例
        
    Raises:
        ValueError: 如果模型类型不支持
    """
    model_type = model_type.lower()
    
    if model_type == 'pls':
        return PLSModel(**kwargs)
    elif model_type == 'scca':
        return SparseCCAModel(**kwargs)
    elif model_type == 'adaptive_pls':
        return AdaptivePLSModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'pls', 'scca', 'adaptive_pls'")


class AdaptivePLSModel(BaseBrainBehaviorModel):
    """
    自适应PLS模型 - 使用内部交叉验证确定最优成分数量
    """
    
    def __init__(self, n_components_range: list = None, cv_folds: int = 5, 
                 criterion: str = 'canonical_correlation', random_state: Optional[int] = None,
                 scale: bool = True, max_iter: int = 500, tol: float = 1e-06):
        """
        初始化自适应PLS模型
        
        Args:
            n_components_range: 成分数量搜索范围，默认 [1, 2, 3, 4, 5]
            cv_folds: 内部交叉验证折数
            criterion: 选择标准 ('canonical_correlation', 'variance_explained', 'stability')
            random_state: 随机种子
            scale: 是否标准化数据
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        if n_components_range is None:
            n_components_range = [1, 2, 3, 4, 5]
        
        self.n_components_range = n_components_range
        self.cv_folds = cv_folds
        self.criterion = criterion
        self.random_state = random_state
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        
        self.optimal_n_components = None
        self.cv_results_ = None
        self.model = None
        self.is_fitted = False
        self.n_components = None  # 用于兼容性
        
    def _evaluate_n_components(self, X: np.ndarray, Y: np.ndarray, 
                              n_components: int) -> Dict[str, float]:
        """
        评估特定成分数量的性能
        
        Args:
            X: 脑数据
            Y: 行为数据
            n_components: 成分数量
            
        Returns:
            评估指标字典
        """
        # 创建内部交叉验证
        kf = KFold(n_splits=self.cv_folds, shuffle=True, 
                  random_state=self.random_state)
        
        canonical_corrs = []
        var_exp_X_list = []
        var_exp_Y_list = []
        stability_scores = []
        
        for train_idx, val_idx in kf.split(X):
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # 创建并拟合模型
            pls_model = PLSCanonical(
                n_components=n_components,
                scale=self.scale,
                max_iter=self.max_iter,
                tol=self.tol
            )
            
            pls_model.fit(X_train, Y_train)
            
            # 转换验证集
            X_val_scores, Y_val_scores = pls_model.transform(X_val, Y_val)
            
            # 计算典型相关系数
            corrs = []
            for i in range(n_components):
                corr, _ = pearsonr(X_val_scores[:, i], Y_val_scores[:, i])
                corrs.append(corr)
            canonical_corrs.append(np.mean(corrs))
            
            # 计算方差解释（简化版）
            var_exp_X_list.append(np.var(X_val_scores, axis=0).sum() / 
                                np.var(X_val, axis=0).sum() * 100)
            var_exp_Y_list.append(np.var(Y_val_scores, axis=0).sum() / 
                                np.var(Y_val, axis=0).sum() * 100)
            
            # 计算稳定性（载荷的一致性）
            if hasattr(pls_model, 'x_loadings_'):
                loadings = pls_model.x_loadings_
                # 计算载荷的稀疏性和稳定性
                stability = np.mean(np.abs(loadings)) / (np.std(loadings) + 1e-8)
                stability_scores.append(stability)
        
        return {
            'canonical_correlation': np.mean(canonical_corrs),
            'variance_explained_X': np.mean(var_exp_X_list),
            'variance_explained_Y': np.mean(var_exp_Y_list),
            'stability': np.mean(stability_scores) if stability_scores else 0.0,
            'canonical_correlation_std': np.std(canonical_corrs)
        }
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame]) -> 'AdaptivePLSModel':
        """
        拟合自适应PLS模型 - 自动选择最优成分数量
        
        Args:
            X: 脑数据 (n_samples, n_brain_features)
            Y: 行为数据 (n_samples, n_behavioral_features)
            
        Returns:
            self
        """
        logger.info(f"Starting adaptive PLS fitting with component range: {self.n_components_range}")
        logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        # 评估每个成分数量
        cv_results = {}
        best_score = -np.inf
        best_n_components = 1
        
        for n_comp in self.n_components_range:
            logger.info(f"Evaluating n_components = {n_comp}")
            metrics = self._evaluate_n_components(X_values, Y_values, n_comp)
            cv_results[n_comp] = metrics
            
            # 根据选择标准选择最优成分数量
            score = metrics[self.criterion]
            if score > best_score:
                best_score = score
                best_n_components = n_comp
        
        self.optimal_n_components = best_n_components
        self.cv_results_ = cv_results
        self.n_components = best_n_components  # 更新n_components用于兼容性
        
        logger.info(f"Optimal n_components selected: {self.optimal_n_components} (criterion: {self.criterion})")
        
        # 使用最优成分数量创建最终模型
        self.model = PLSCanonical(
            n_components=self.optimal_n_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        # 拟合最终模型
        self.model.fit(X_values, Y_values)
        self.is_fitted = True
        
        logger.info("Adaptive PLS model fitting completed")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                  Y: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据到潜变量空间
        
        Args:
            X: 脑数据
            Y: 行为数据
            
        Returns:
            X_scores: 脑潜变量分数
            Y_scores: 行为潜变量分数
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        return self.model.transform(X, Y)
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings: 脑载荷矩阵
            Y_loadings: 行为载荷矩阵
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        return self.model.x_loadings_, self.model.y_loadings_
    
    def get_model_info(self) -> Dict[str, Union[int, bool, float, str]]:
        """
        获取模型信息
        
        Returns:
            模型参数字典
        """
        return {
            'model_type': 'Adaptive-PLS',
            'n_components_range': self.n_components_range,
            'optimal_n_components': self.optimal_n_components,
            'cv_folds': self.cv_folds,
            'criterion': self.criterion,
            'scale': self.scale,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'is_fitted': self.is_fitted
        }
    
    def get_cv_results(self) -> Dict[int, Dict[str, float]]:
        """
        获取交叉验证结果
        
        Returns:
            各成分数量的评估结果
        """
        return self.cv_results_


def get_available_models() -> list:
    """
    获取可用的模型类型
    
    Returns:
        可用模型类型列表
    """
    return ['pls', 'scca', 'adaptive_pls']