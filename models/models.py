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


_SCCA_IPLS = None
_scca_ipls_uses_latent_dimensions = False
_cca_zoo_import_error = None

try:
    from cca_zoo.linear import SCCA_IPLS as _SCCA_IPLS  # type: ignore
    _scca_ipls_uses_latent_dimensions = True
except ImportError as e_linear:
    try:
        from cca_zoo.models import SCCA_IPLS as _SCCA_IPLS  # type: ignore
        _scca_ipls_uses_latent_dimensions = False
    except ImportError as e_models:
        _SCCA_IPLS = None
        _cca_zoo_import_error = (e_linear, e_models)


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
    
    def get_params(self, deep: bool = True):
        """
        获取模型参数（用于交叉验证中的模型复制）
        
        Args:
            deep: 是否深拷贝
            
        Returns:
            参数字典
        """
        return {
            'n_components': self.n_components,
            'random_state': self.random_state
        }
        
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
            tol=tol
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
    
    def get_params(self, deep: bool = True):
        """
        获取模型参数（用于交叉验证中的模型复制）
        
        Args:
            deep: 是否深拷贝
            
        Returns:
            参数字典
        """
        return {
            'n_components': self.n_components,
            'scale': self.scale,
            'max_iter': self.max_iter,
            'tol': self.tol
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

        if _SCCA_IPLS is None:
            raise ImportError(
                "SparseCCAModel requires the 'cca-zoo' library with the SCCA_IPLS "
                "implementation (tested with cca-zoo>=2.0). "
                "Please install or upgrade it with: pip install -U cca-zoo"
            )

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        if _scca_ipls_uses_latent_dimensions:
            self.model = _SCCA_IPLS(
                latent_dimensions=self.n_components,
                random_state=self.random_state,
                tol=self.tol,
                epochs=self.max_iter,
            )
        else:
            self.model = _SCCA_IPLS(
                latent_dims=self.n_components,
                random_state=self.random_state,
                c=[self.sparsity_X, self.sparsity_Y],
                max_iter=self.max_iter,
                tol=self.tol,
            )
    
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
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y

        X_scaled = self.scaler_X.fit_transform(X_values)
        Y_scaled = self.scaler_Y.fit_transform(Y_values)

        self.model.fit([X_scaled, Y_scaled])
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
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y

        X_scaled = self.scaler_X.transform(X_values)
        Y_scaled = self.scaler_Y.transform(Y_values)

        X_scores, Y_scores = self.model.transform([X_scaled, Y_scaled])
        return X_scores, Y_scores
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings, Y_loadings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")

        if hasattr(self.model, "weights"):
            X_loadings, Y_loadings = self.model.weights
        elif hasattr(self.model, "weights_"):
            X_loadings, Y_loadings = self.model.weights_
        else:
            raise AttributeError("Underlying SCCA_IPLS model does not expose weights")

        return X_loadings, Y_loadings
    
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
            'implementation': 'cca-zoo SCCA_IPLS'
        }


def create_model(model_type: str, **kwargs) -> BaseBrainBehaviorModel:
    """
    工厂函数 - 创建模型实例
    
    Args:
        model_type: 模型类型 ('pls', 'scca', 'adaptive_pls', 'adaptive_scca')
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
    elif model_type == 'adaptive_scca':
        return AdaptiveSCCAModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'pls', 'scca', 'adaptive_pls', 'adaptive_scca'")


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
    
    def get_params(self, deep: bool = True):
        """
        获取模型参数（用于交叉验证中的模型复制）
        
        Args:
            deep: 是否深拷贝
            
        Returns:
            参数字典
        """
        return {
            'n_components_range': self.n_components_range,
            'cv_folds': self.cv_folds,
            'criterion': self.criterion,
            'scale': self.scale,
            'max_iter': self.max_iter,
            'tol': self.tol
        }


class AdaptiveSCCAModel(BaseBrainBehaviorModel):
    """
    自适应 Sparse-CCA 模型 - 使用内部交叉验证同时确定最优成分数和稀疏度参数
    """
    
    def __init__(self, n_components_range: list = None,
                 sparsity_X_range: list = None, 
                 sparsity_Y_range: list = None,
                 cv_folds: int = 5, 
                 criterion: str = 'canonical_correlation',
                 random_state: Optional[int] = None,
                 max_iter: int = 1000, 
                 tol: float = 1e-06):
        """
        初始化自适应 Sparse-CCA 模型
        
        Args:
            n_components_range: 成分数量搜索范围，默认 [2, 3, 4, 5]
            sparsity_X_range: 脑数据稀疏度搜索范围，默认 [0.001, 0.005, 0.01, 0.05]
            sparsity_Y_range: 行为数据稀疏度搜索范围，默认 [0.1, 0.2, 0.3]
            cv_folds: 内部交叉验证折数
            criterion: 选择标准 ('canonical_correlation', 'variance_explained')
            random_state: 随机种子
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        # 默认搜索范围
        if n_components_range is None:
            n_components_range = [2, 3, 4, 5]
        if sparsity_X_range is None:
            sparsity_X_range = [0.001, 0.005, 0.01, 0.05]  # 减少搜索点以控制计算量
        if sparsity_Y_range is None:
            sparsity_Y_range = [0.1, 0.2, 0.3]  # 减少搜索点以控制计算量
        
        # 初始化基类（使用最大成分数）
        super().__init__(max(n_components_range), random_state)
        
        self.n_components_range = n_components_range
        self.sparsity_X_range = sparsity_X_range
        self.sparsity_Y_range = sparsity_Y_range
        self.cv_folds = cv_folds
        self.criterion = criterion
        self.max_iter = max_iter
        self.tol = tol
        
        self.optimal_n_components = None
        self.optimal_sparsity_X = None
        self.optimal_sparsity_Y = None
        self.cv_results_ = None
        self.model = None
        self.is_fitted = False
        
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # 检查 cca-zoo 是否可用
        if _SCCA_IPLS is None:
            raise ImportError(
                "AdaptiveSCCAModel requires the 'cca-zoo' library with the SCCA_IPLS "
                "implementation. Please install it with: pip install -U cca-zoo"
            )
    
    def _evaluate_hyperparameter_combo(self, X: np.ndarray, Y: np.ndarray, 
                                        n_components: int,
                                        sparsity_X: float, sparsity_Y: float) -> Dict[str, float]:
        """
        评估特定超参数组合的性能
        
        Args:
            X: 脑数据
            Y: 行为数据
            n_components: 成分数量
            sparsity_X: 脑数据稀疏度
            sparsity_Y: 行为数据稀疏度
            
        Returns:
            评估指标字典
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, 
                  random_state=self.random_state)
        
        canonical_corrs = []
        var_exp_X_list = []
        var_exp_Y_list = []
        
        for train_idx, val_idx in kf.split(X):
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # 标准化
            scaler_X_fold = StandardScaler()
            scaler_Y_fold = StandardScaler()
            X_train_scaled = scaler_X_fold.fit_transform(X_train)
            Y_train_scaled = scaler_Y_fold.fit_transform(Y_train)
            X_val_scaled = scaler_X_fold.transform(X_val)
            Y_val_scaled = scaler_Y_fold.transform(Y_val)
            
            # 创建并拟合 SCCA 模型
            try:
                if _scca_ipls_uses_latent_dimensions:
                    scca_model = _SCCA_IPLS(
                        latent_dimensions=n_components,
                        random_state=self.random_state,
                        tol=self.tol,
                        epochs=self.max_iter,
                    )
                else:
                    scca_model = _SCCA_IPLS(
                        latent_dims=n_components,
                        random_state=self.random_state,
                        c=[sparsity_X, sparsity_Y],
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
                
                scca_model.fit([X_train_scaled, Y_train_scaled])
                
                # 转换验证集
                X_val_scores, Y_val_scores = scca_model.transform([X_val_scaled, Y_val_scaled])
                
                # 计算典型相关系数
                corrs = []
                for i in range(n_components):
                    if X_val_scores.shape[1] > i and Y_val_scores.shape[1] > i:
                        corr, _ = pearsonr(X_val_scores[:, i], Y_val_scores[:, i])
                        if not np.isnan(corr):
                            corrs.append(corr)
                
                if corrs:
                    canonical_corrs.append(np.mean(corrs))
                    
                    # 计算方差解释
                    var_exp_X = np.var(X_val_scores, axis=0).sum() / np.var(X_val_scaled, axis=0).sum() * 100
                    var_exp_Y = np.var(Y_val_scores, axis=0).sum() / np.var(Y_val_scaled, axis=0).sum() * 100
                    var_exp_X_list.append(var_exp_X)
                    var_exp_Y_list.append(var_exp_Y)
                    
            except Exception as e:
                logger.warning(f"SCCA fitting failed for n_comp={n_components}, sparsity_X={sparsity_X}, sparsity_Y={sparsity_Y}: {e}")
                continue
        
        if not canonical_corrs:
            return {
                'canonical_correlation': -np.inf,
                'variance_explained_X': 0.0,
                'variance_explained_Y': 0.0,
                'canonical_correlation_std': np.inf
            }
        
        return {
            'canonical_correlation': np.mean(canonical_corrs),
            'variance_explained_X': np.mean(var_exp_X_list) if var_exp_X_list else 0.0,
            'variance_explained_Y': np.mean(var_exp_Y_list) if var_exp_Y_list else 0.0,
            'canonical_correlation_std': np.std(canonical_corrs)
        }
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame]) -> 'AdaptiveSCCAModel':
        """
        拟合自适应 Sparse-CCA 模型 - 自动选择最优成分数和稀疏度参数
        
        Args:
            X: 脑数据 (n_samples, n_brain_features)
            Y: 行为数据 (n_samples, n_behavioral_features)
            
        Returns:
            self
        """
        logger.info(f"Starting adaptive SCCA fitting with n_components range: {self.n_components_range}")
        logger.info(f"sparsity_X range: {self.sparsity_X_range}")
        logger.info(f"sparsity_Y range: {self.sparsity_Y_range}")
        logger.info(f"Total combinations to evaluate: {len(self.n_components_range) * len(self.sparsity_X_range) * len(self.sparsity_Y_range)}")
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
        
        # 网格搜索最优超参数组合
        cv_results = {}
        best_score = -np.inf
        best_n_components = self.n_components_range[0]
        best_sparsity_X = self.sparsity_X_range[0]
        best_sparsity_Y = self.sparsity_Y_range[0]
        
        for n_comp in self.n_components_range:
            for sparsity_X in self.sparsity_X_range:
                for sparsity_Y in self.sparsity_Y_range:
                    logger.info(f"Evaluating n_comp={n_comp}, sparsity_X={sparsity_X}, sparsity_Y={sparsity_Y}")
                    metrics = self._evaluate_hyperparameter_combo(X_values, Y_values, n_comp, sparsity_X, sparsity_Y)
                    cv_results[(n_comp, sparsity_X, sparsity_Y)] = metrics
                    
                    # 根据选择标准选择最优参数
                    score = metrics[self.criterion]
                    if score > best_score:
                        best_score = score
                        best_n_components = n_comp
                        best_sparsity_X = sparsity_X
                        best_sparsity_Y = sparsity_Y
        
        self.optimal_n_components = best_n_components
        self.optimal_sparsity_X = best_sparsity_X
        self.optimal_sparsity_Y = best_sparsity_Y
        self.n_components = best_n_components  # 更新 n_components 用于兼容性
        self.cv_results_ = cv_results
        
        logger.info(f"Optimal hyperparameters selected:")
        logger.info(f"  n_components={self.optimal_n_components}")
        logger.info(f"  sparsity_X={self.optimal_sparsity_X}")
        logger.info(f"  sparsity_Y={self.optimal_sparsity_Y}")
        logger.info(f"  Best {self.criterion}: {best_score:.4f}")
        
        # 使用最优超参数创建最终模型
        if _scca_ipls_uses_latent_dimensions:
            self.model = _SCCA_IPLS(
                latent_dimensions=self.optimal_n_components,
                random_state=self.random_state,
                tol=self.tol,
                epochs=self.max_iter,
            )
        else:
            self.model = _SCCA_IPLS(
                latent_dims=self.optimal_n_components,
                random_state=self.random_state,
                c=[self.optimal_sparsity_X, self.optimal_sparsity_Y],
                max_iter=self.max_iter,
                tol=self.tol,
            )
        
        # 标准化并拟合最终模型
        X_scaled = self.scaler_X.fit_transform(X_values)
        Y_scaled = self.scaler_Y.fit_transform(Y_values)
        self.model.fit([X_scaled, Y_scaled])
        self.is_fitted = True
        
        logger.info("Adaptive SCCA model fitting completed")
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
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(Y, pd.DataFrame):
            Y_values = Y.values
        else:
            Y_values = Y
        
        X_scaled = self.scaler_X.transform(X_values)
        Y_scaled = self.scaler_Y.transform(Y_values)
        
        X_scores, Y_scores = self.model.transform([X_scaled, Y_scaled])
        return X_scores, Y_scores
    
    def get_loadings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取载荷矩阵
        
        Returns:
            X_loadings: 脑载荷矩阵
            Y_loadings: 行为载荷矩阵
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        if hasattr(self.model, "weights"):
            X_loadings, Y_loadings = self.model.weights
        elif hasattr(self.model, "weights_"):
            X_loadings, Y_loadings = self.model.weights_
        else:
            raise AttributeError("Underlying SCCA_IPLS model does not expose weights")
        
        return X_loadings, Y_loadings
    
    def get_model_info(self) -> Dict[str, Union[int, bool, float, str, list]]:
        """
        获取模型信息
        
        Returns:
            模型参数字典
        """
        return {
            'model_type': 'Adaptive-SCCA',
            'n_components_range': self.n_components_range,
            'optimal_n_components': self.optimal_n_components,
            'sparsity_X_range': self.sparsity_X_range,
            'sparsity_Y_range': self.sparsity_Y_range,
            'optimal_sparsity_X': self.optimal_sparsity_X,
            'optimal_sparsity_Y': self.optimal_sparsity_Y,
            'cv_folds': self.cv_folds,
            'criterion': self.criterion,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'is_fitted': self.is_fitted
        }
    
    def get_cv_results(self) -> Dict[Tuple[int, float, float], Dict[str, float]]:
        """
        获取交叉验证结果
        
        Returns:
            各超参数组合的评估结果 (n_components, sparsity_X, sparsity_Y) -> metrics
        """
        return self.cv_results_
    
    def get_params(self, deep: bool = True):
        """
        获取模型参数（用于交叉验证中的模型复制）
        
        Args:
            deep: 是否深拷贝
            
        Returns:
            参数字典
        """
        return {
            'n_components_range': self.n_components_range,
            'sparsity_X_range': self.sparsity_X_range,
            'sparsity_Y_range': self.sparsity_Y_range,
            'cv_folds': self.cv_folds,
            'criterion': self.criterion,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state
        }


def get_available_models() -> list:
    """
    获取可用的模型类型
    
    Returns:
        可用模型类型列表
    """
    return ['pls', 'scca', 'adaptive_pls', 'adaptive_scca']
