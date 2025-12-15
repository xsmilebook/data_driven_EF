#!/usr/bin/env python3
"""
数据加载模块 - 负责读取脑影像和行为数据
支持 EFNY 数据集的标准格式
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EFNYDataLoader:
    """EFNY 数据集加载器"""
    
    def __init__(self, data_root: Union[str, Path] = "/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY"):
        """
        初始化数据加载器
        
        Args:
            data_root: EFNY 数据根目录
        """
        self.data_root = Path(data_root)
        self.brain_data_path = self.data_root / "fc_vector" / "Schaefer100" / "EFNY_Schaefer100_FC_matrix.npy"
        self.behavioral_data_path = self.data_root / "table" / "demo" / "EFNY_behavioral_data.csv"
        self.sublist_path = self.data_root / "table" / "sublist" / "sublist.txt"
        
    def load_brain_data(self) -> np.ndarray:
        """
        加载脑功能连接数据
        
        Returns:
            brain_data: 形状为 (n_subjects, n_features) 的功能连接矩阵
            
        Raises:
            FileNotFoundError: 如果脑数据文件不存在
            ValueError: 如果数据格式不正确
        """
        logger.info(f"Loading brain data from: {self.brain_data_path}")
        
        if not self.brain_data_path.exists():
            raise FileNotFoundError(f"Brain data file not found: {self.brain_data_path}")
            
        try:
            brain_data = np.load(self.brain_data_path)
            logger.info(f"Brain data loaded successfully: shape {brain_data.shape}")
            return brain_data
        except Exception as e:
            raise ValueError(f"Error loading brain data: {e}")
    
    def load_behavioral_data(self) -> pd.DataFrame:
        """
        加载行为数据
        
        Returns:
            behavioral_data: 包含所有行为指标的数据框
            
        Raises:
            FileNotFoundError: 如果行为数据文件不存在
            ValueError: 如果数据格式不正确
        """
        logger.info(f"Loading behavioral data from: {self.behavioral_data_path}")
        
        if not self.behavioral_data_path.exists():
            raise FileNotFoundError(f"Behavioral data file not found: {self.behavioral_data_path}")
            
        try:
            behavioral_data = pd.read_csv(self.behavioral_data_path)
            logger.info(f"Behavioral data loaded successfully: shape {behavioral_data.shape}")
            logger.info(f"Available behavioral measures: {list(behavioral_data.columns)}")
            return behavioral_data
        except Exception as e:
            raise ValueError(f"Error loading behavioral data: {e}")
    
    def load_subject_list(self) -> np.ndarray:
        """
        加载被试列表
        
        Returns:
            subject_ids: 被试ID数组
            
        Raises:
            FileNotFoundError: 如果被试列表文件不存在
        """
        logger.info(f"Loading subject list from: {self.sublist_path}")
        
        if not self.sublist_path.exists():
            raise FileNotFoundError(f"Subject list file not found: {self.sublist_path}")
            
        try:
            subject_ids = np.loadtxt(self.sublist_path, dtype=str)
            logger.info(f"Subject list loaded successfully: {len(subject_ids)} subjects")
            return subject_ids
        except Exception as e:
            raise ValueError(f"Error loading subject list: {e}")
    
    def load_all_data(self) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        加载所有数据
        
        Returns:
            brain_data: 脑功能连接数据 (n_subjects, n_features)
            behavioral_data: 行为数据 (n_subjects, n_measures)
            subject_ids: 被试ID数组 (n_subjects,)
            
        Raises:
            ValueError: 如果数据维度不匹配
        """
        brain_data = self.load_brain_data()
        behavioral_data = self.load_behavioral_data()
        subject_ids = self.load_subject_list()
        
        # 验证数据维度一致性
        n_subjects_brain = brain_data.shape[0]
        n_subjects_behavior = len(behavioral_data)
        n_subjects_list = len(subject_ids)
        
        if not (n_subjects_brain == n_subjects_behavior == n_subjects_list):
            raise ValueError(
                f"Data dimension mismatch: "
                f"brain={n_subjects_brain}, "
                f"behavior={n_subjects_behavior}, "
                f"subjects={n_subjects_list}"
            )
        
        logger.info(f"All data loaded successfully. Total subjects: {n_subjects_brain}")
        return brain_data, behavioral_data, subject_ids
    
    def get_available_behavioral_measures(self) -> list:
        """
        获取可用的行为指标列表
        
        Returns:
            行为指标名称列表
        """
        try:
            behavioral_data = self.load_behavioral_data()
            return list(behavioral_data.columns)
        except Exception as e:
            logger.warning(f"Could not load behavioral measures: {e}")
            return []
    
    def filter_subjects_by_data_quality(self, brain_data: np.ndarray, 
                                       behavioral_data: pd.DataFrame,
                                       subject_ids: np.ndarray,
                                       max_missing_rate: float = 0.1) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        基于数据质量过滤被试
        
        Args:
            brain_data: 脑数据
            behavioral_data: 行为数据
            subject_ids: 被试ID
            max_missing_rate: 最大缺失率阈值
            
        Returns:
            过滤后的数据
        """
        logger.info(f"Filtering subjects with max missing rate: {max_missing_rate}")
        
        # 检查脑数据中的NaN值
        brain_nan_rate = np.isnan(brain_data).sum(axis=1) / brain_data.shape[1]
        valid_brain_mask = brain_nan_rate <= max_missing_rate
        
        # 检查行为数据中的NaN值
        behavioral_nan_rate = behavioral_data.isnull().sum(axis=1) / len(behavioral_data.columns)
        valid_behavioral_mask = behavioral_nan_rate <= max_missing_rate
        
        # 合并有效掩码
        valid_mask = valid_brain_mask & valid_behavioral_mask.values
        
        n_original = len(subject_ids)
        n_filtered = valid_mask.sum()
        
        logger.info(f"Filtered subjects: {n_original} -> {n_filtered} ({n_original-n_filtered} removed)")
        
        if n_filtered == 0:
            raise ValueError("No subjects passed quality filtering")
        
        return brain_data[valid_mask], behavioral_data[valid_mask], subject_ids[valid_mask]


def create_synthetic_data(n_subjects: int = 200, 
                          n_brain_features: Optional[int] = None,
                          n_behavioral_measures: int = 30,
                          atlas: str = "schaefer100",
                          random_state: Optional[int] = 42) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    创建合成数据用于测试
    
    Args:
        n_subjects: 被试数量
        n_brain_features: 脑特征数量（若为None则根据atlas自动推断）
        n_behavioral_measures: 行为指标数量
        atlas: 脑图谱名称（例如"schaefer100"或"schaefer400"）
        random_state: 随机种子
        
    Returns:
        brain_data: 合成脑数据
        behavioral_data: 合成行为数据
        covariates: 合成协变量
    """
    if n_brain_features is None:
        atlas_lower = atlas.lower()
        if atlas_lower == "schaefer100":
            n_brain_features = 100 * 99 // 2
        elif atlas_lower == "schaefer400":
            n_brain_features = 400 * 399 // 2
        else:
            raise ValueError(f"Unsupported atlas for synthetic data: {atlas}")

    logger.info(f"Creating synthetic data: {n_subjects} subjects, {n_brain_features} features (atlas={atlas})")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # 创建潜在因子
    n_latent = 3
    latent_factors = np.random.normal(0, 1, size=(n_subjects, n_latent))
    
    # 脑数据
    brain_loadings = np.random.normal(0, 0.3, size=(n_latent, n_brain_features))
    brain_data = latent_factors @ brain_loadings + np.random.normal(0, 0.5, size=(n_subjects, n_brain_features))
    
    # 行为数据
    behavioral_metrics = [
        'SST_SSRT', 'CPT_d_prime', 'CPT_ACC', 'CPT_Reaction_Time',
        'FLANKER_Contrast_ACC', 'FLANKER_Contrast_RT', 'ColorStroop_Contrast_ACC',
        'ColorStroop_Contrast_RT', 'EmotionStroop_Contrast_ACC', 'EmotionStroop_Contrast_RT',
        'Number1Back_ACC', 'Number1Back_Reaction_Time', 'Number2Back_ACC',
        'Number2Back_Reaction_Time', 'Spatial1Back_ACC', 'Spatial1Back_Reaction_Time',
        'Spatial2Back_ACC', 'Spatial2Back_Reaction_Time', 'Emotion1Back_ACC',
        'Emotion1Back_Reaction_Time', 'Emotion2Back_ACC', 'Emotion2Back_Reaction_Time',
        'KT_ACC', 'DCCS_ACC', 'DCCS_Reaction_Time', 'DCCS_Switch_Cost',
        'EmotionSwitch_ACC', 'EmotionSwitch_Reaction_Time', 'DT_ACC',
        'DT_Reaction_Time', 'DT_Switch_Cost'
    ]
    
    selected_metrics = behavioral_metrics[:n_behavioral_measures]
    behavioral_loadings = np.random.normal(0, 0.4, size=(n_latent, n_behavioral_measures))
    behavioral_data = latent_factors @ behavioral_loadings + np.random.normal(0, 0.3, size=(n_subjects, n_behavioral_measures))
    
    # 协变量（EFNY标准：age, sex, meanFD）
    age = np.random.normal(25, 5, size=n_subjects)
    sex = np.random.choice([0, 1], size=n_subjects)
    meanFD = np.random.normal(0.15, 0.05, size=n_subjects)
    covariates = pd.DataFrame({'age': age, 'sex': sex, 'meanFD': meanFD})
    
    # 添加协变量效应
    brain_data += np.outer(age, np.random.normal(0, 0.1, n_brain_features))
    brain_data += np.outer(sex, np.random.normal(0, 0.2, n_brain_features))
    brain_data += np.outer(meanFD, np.random.normal(0, 2, n_brain_features))
    behavioral_data += np.outer(age, np.random.normal(0, 0.05, n_behavioral_measures))
    behavioral_data += np.outer(sex, np.random.normal(0, 0.1, n_behavioral_measures))
    behavioral_data += np.outer(meanFD, np.random.normal(0, 1, n_behavioral_measures))
    
    brain_df = pd.DataFrame(brain_data, columns=[f'FC_{i}' for i in range(n_brain_features)])
    behavioral_df = pd.DataFrame(behavioral_data, columns=selected_metrics)
    
    logger.info("Synthetic data created successfully")
    return brain_df, behavioral_df, covariates
