#!/usr/bin/env python3
"""
预处理模块 - 混杂变量回归
遵循 sklearn Transformer 标准，防止数据泄露
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """
    混杂变量回归器 - 遵循 sklearn Transformer 标准
    
    该类用于从目标变量中回归出混杂变量的影响，
    确保在交叉验证中只在训练集上拟合回归系数，
    然后应用到测试集，防止数据泄露。
    
    Attributes:
        standardize_: 是否标准化残差
        regression_models_: 每个特征的回归模型
        scalers_: 每个特征的标准化器
    """
    
    def __init__(self, standardize: bool = True, copy: bool = True):
        """
        初始化混杂变量回归器
        
        Args:
            standardize: 是否标准化残差（均值为0，标准差为1）
            copy: 是否在转换时复制数据
        """
        self.standardize = standardize
        self.copy = copy
        self.regression_models_ = {}
        self.scalers_ = {}
        self.feature_names_ = None
        self.confound_names_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'ConfoundRegressor':
        """
        拟合混杂变量回归模型
        
        Args:
            X: 目标数据 (n_samples, n_features)
            y: 未使用，为了兼容 sklearn 接口
            confounds: 混杂变量 (n_samples, n_confounds)
            
        Returns:
            self: 返回拟合后的实例
            
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        logger.info(f"Fitting ConfoundRegressor with {X.shape[1]} features and {confounds.shape[1]} confounds")
        
        # 转换输入数据
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(confounds, pd.DataFrame):
            self.confound_names_ = confounds.columns.tolist()
            confounds_values = confounds.values
        else:
            confounds_values = confounds
            
        n_samples, n_features = X_values.shape
        n_confounds = confounds_values.shape[1]
        
        if n_samples != confounds_values.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {n_samples} samples, confounds has {confounds_values.shape[0]}")
        
        # 为每个特征拟合单独的回归模型
        self.regression_models_ = {}
        self.scalers_ = {}
        
        for i in range(n_features):
            # 拟合线性回归模型
            lr = LinearRegression()
            lr.fit(confounds_values, X_values[:, i])
            self.regression_models_[i] = lr
            
            # 如果需要标准化，计算残差的均值和标准差
            if self.standardize:
                # 计算残差
                predicted = lr.predict(confounds_values)
                residuals = X_values[:, i] - predicted
                
                # 计算标准化参数
                mean_resid = np.mean(residuals)
                std_resid = np.std(residuals)
                
                # 避免除零
                if std_resid == 0:
                    std_resid = 1.0
                    logger.warning(f"Feature {i} has zero standard deviation in residuals")
                
                self.scalers_[i] = {'mean': mean_resid, 'scale': std_resid}
        
        logger.info("ConfoundRegressor fitting completed")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame],
                  confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        转换数据 - 回归出混杂变量影响
        
        Args:
            X: 目标数据 (n_samples, n_features)
            confounds: 混杂变量 (n_samples, n_confounds)
            
        Returns:
            转换后的数据，保持与输入相同的类型
        """
        logger.info(f"Transforming data with {X.shape[0]} samples and {X.shape[1]} features")
        
        # 转换输入数据
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            return_as_df = True
        else:
            X_values = X
            return_as_df = False
            
        if isinstance(confounds, pd.DataFrame):
            confounds_values = confounds.values
        else:
            confounds_values = confounds
        
        # 复制数据以避免修改原始数据
        if self.copy:
            X_values = X_values.copy()
        
        n_samples, n_features = X_values.shape
        
        # 为每个特征回归混杂变量
        residuals = np.zeros_like(X_values)
        
        for i in range(n_features):
            if i in self.regression_models_:
                lr = self.regression_models_[i]
                
                # 预测混杂变量效应
                predicted = lr.predict(confounds_values)
                
                # 计算残差
                residual = X_values[:, i] - predicted
                
                # 如果需要标准化
                if self.standardize and i in self.scalers_:
                    mean_resid = self.scalers_[i]['mean']
                    std_resid = self.scalers_[i]['scale']
                    residual = (residual - mean_resid) / std_resid
                
                residuals[:, i] = residual
            else:
                # 如果没有模型，保持原始数据
                residuals[:, i] = X_values[:, i]
                logger.warning(f"No regression model for feature {i}, keeping original data")
        
        logger.info("ConfoundRegressor transformation completed")
        
        # 返回与输入相同类型的数据
        if return_as_df:
            if self.feature_names_ is not None:
                return pd.DataFrame(residuals, columns=self.feature_names_, index=X.index)
            else:
                return pd.DataFrame(residuals, index=X.index)
        else:
            return residuals
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                      confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        拟合并转换数据
        
        Args:
            X: 目标数据
            y: 未使用
            confounds: 混杂变量
            
        Returns:
            转换后的数据
        """
        return self.fit(X, y, confounds).transform(X, confounds)
    
    def get_feature_names_out(self, input_features=None):
        """
        获取输出特征名称 - 为了兼容 sklearn 管道
        
        Args:
            input_features: 输入特征名称
            
        Returns:
            输出特征名称
        """
        if input_features is not None:
            return input_features
        elif self.feature_names_ is not None:
            return self.feature_names_
        else:
            return None
    
    def get_params(self, deep=True):
        """获取参数 - sklearn 兼容性"""
        return {
            'standardize': self.standardize,
            'copy': self.copy
        }
    
    def set_params(self, **params):
        """设置参数 - sklearn 兼容性"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class DataQualityFilter:
    """数据质量过滤器"""
    
    def __init__(self, max_missing_rate: float = 0.1, min_std: float = 1e-6):
        """
        初始化数据质量过滤器
        
        Args:
            max_missing_rate: 最大缺失率
            min_std: 最小标准差
        """
        self.max_missing_rate = max_missing_rate
        self.min_std = min_std
        
    def filter_features(self, X: Union[np.ndarray, pd.DataFrame],
                         confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        过滤特征 - 移除低质量特征
        
        Args:
            X: 输入数据
            confounds: 混杂变量
            
        Returns:
            过滤后的数据和特征掩码
        """
        logger.info("Filtering low-quality features")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
            return_as_df = True
        else:
            X_values = X
            feature_names = None
            return_as_df = False
        
        n_samples, n_features = X_values.shape
        
        # 计算每个特征的缺失率
        missing_rates = np.isnan(X_values).sum(axis=0) / n_samples
        
        # 计算每个特征的标准差
        std_values = np.nanstd(X_values, axis=0)
        
        # 创建质量掩码
        quality_mask = (missing_rates <= self.max_missing_rate) & (std_values >= self.min_std)
        
        n_original = n_features
        n_filtered = quality_mask.sum()
        
        logger.info(f"Feature filtering: {n_original} -> {n_filtered} ({n_original-n_filtered} removed)")
        
        # 过滤数据
        X_filtered = X_values[:, quality_mask]
        
        if return_as_df and feature_names is not None:
            filtered_names = [feature_names[i] for i in range(n_features) if quality_mask[i]]
            X_filtered = pd.DataFrame(X_filtered, columns=filtered_names, index=X.index)
        
        return X_filtered, quality_mask
    
    def filter_subjects(self, X: Union[np.ndarray, pd.DataFrame],
                       confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        过滤被试 - 移除低质量被试
        
        Args:
            X: 输入数据
            confounds: 混杂变量
            
        Returns:
            过滤后的数据和被试掩码
        """
        logger.info("Filtering low-quality subjects")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            return_as_df = True
        else:
            X_values = X
            return_as_df = False
        
        n_samples, n_features = X_values.shape
        
        # 计算每个被试的缺失率
        missing_rates = np.isnan(X_values).sum(axis=1) / n_features
        
        # 创建被试掩码
        subject_mask = missing_rates <= self.max_missing_rate
        
        n_original = n_samples
        n_filtered = subject_mask.sum()
        
        logger.info(f"Subject filtering: {n_original} -> {n_filtered} ({n_original-n_filtered} removed)")
        
        # 过滤数据
        X_filtered = X_values[subject_mask]
        
        if return_as_df:
            X_filtered = pd.DataFrame(X_filtered, index=X.index[subject_mask])
        
        return X_filtered, subject_mask


def create_preprocessing_pipeline(max_missing_rate: float = 0.1, 
                                standardize_confounds: bool = True) -> Tuple[DataQualityFilter, ConfoundRegressor]:
    """
    创建预处理管道
    
    Args:
        max_missing_rate: 最大缺失率
        standardize_confounds: 是否标准化混杂变量回归结果
        
    Returns:
        质量过滤器和混杂变量回归器
    """
    quality_filter = DataQualityFilter(max_missing_rate=max_missing_rate)
    confound_regressor = ConfoundRegressor(standardize=standardize_confounds)
    
    return quality_filter, confound_regressor