#!/usr/bin/env python3
"""
工具模块 - 日志配置、路径管理和通用工具函数
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json
import numpy as np


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[Union[str, Path]] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为 None 则只输出到控制台
        log_format: 日志格式字符串
        
    Returns:
        配置好的 logger 实例
    """
    # 默认日志格式
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 创建 formatter
    formatter = logging.Formatter(log_format)
    
    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging setup completed. Level: {log_level}")
    return logger


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 假设项目根目录是包含 src 的目录
    current_file = Path(__file__).resolve()
    
    # 向上查找直到找到包含 src 的目录
    for parent in current_file.parents:
        if (parent / "src").exists():
            return parent
    
    # 如果没有找到，返回当前文件的父目录
    return current_file.parent


def get_data_dir() -> Path:
    """
    获取数据目录
    
    Returns:
        数据目录路径
    """
    project_root = get_project_root()
    data_dir = project_root / "data" / "EFNY"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_results_dir() -> Path:
    """
    获取结果保存目录
    
    Returns:
        结果目录路径
    """
    project_root = get_project_root()
    results_dir = project_root / "results" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def create_timestamp() -> str:
    """
    创建时间戳字符串
    
    Returns:
        格式为 YYYYMMDD_HHMMSS 的时间戳
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results: Dict[str, Any], 
                output_path: Union[str, Path],
                format: str = "both") -> Dict[str, Path]:
    """
    保存结果到文件
    
    Args:
        results: 要保存的结果字典
        output_path: 输出文件路径（不含扩展名）
        format: 保存格式 ("json", "npz", "both")
        
    Returns:
        保存的文件路径字典
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        
        # 转换 numpy 数组为列表以便 JSON 序列化
        json_results = _convert_numpy_to_list(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        saved_files['json'] = json_path
        logger = logging.getLogger(__name__)
        logger.info(f"Results saved to JSON: {json_path}")
    
    if format in ["npz", "both"]:
        npz_path = output_path.with_suffix(".npz")
        
        # 提取 numpy 数组并保存
        np_arrays = _extract_numpy_arrays(results)
        np.savez_compressed(npz_path, **np_arrays)
        
        saved_files['npz'] = npz_path
        logger = logging.getLogger(__name__)
        logger.info(f"Results saved to NPZ: {npz_path}")
    
    return saved_files


def _convert_numpy_to_list(obj: Any) -> Any:
    """
    递归转换 numpy 数组为 Python 列表
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_list(item) for item in obj)
    else:
        return obj


def _extract_numpy_arrays(obj: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    从字典中提取 numpy 数组
    
    Args:
        obj: 输入字典
        
    Returns:
        只包含 numpy 数组的字典
    """
    np_arrays = {}
    
    for key, value in obj.items():
        if isinstance(value, np.ndarray):
            np_arrays[key] = value
        elif isinstance(value, (int, float, bool, str)):
            # 将标量转换为 0-D numpy 数组
            np_arrays[key] = np.array(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            # 将数值列表转换为 numpy 数组
            np_arrays[key] = np.array(value)
    
    return np_arrays


def load_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载结果
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        加载的结果字典
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix == '.npz':
        npz_file = np.load(file_path)
        results = {}
        for key in npz_file.files:
            results[key] = npz_file[key]
        return results
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def validate_data_shapes(X: np.ndarray, Y: np.ndarray, 
                         confounds: Optional[np.ndarray] = None) -> bool:
    """
    验证数据形状是否一致
    
    Args:
        X: 脑数据 (n_samples, n_features)
        Y: 行为数据 (n_samples, n_targets)
        confounds: 混杂变量 (n_samples, n_confounds)
        
    Returns:
        是否通过验证
        
    Raises:
        ValueError: 如果数据形状不一致
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    
    if n_samples_X != n_samples_Y:
        raise ValueError(
            f"Sample count mismatch: X has {n_samples_X} samples, "
            f"Y has {n_samples_Y} samples"
        )
    
    if confounds is not None:
        n_samples_confounds = confounds.shape[0]
        if n_samples_X != n_samples_confounds:
            raise ValueError(
                f"Sample count mismatch: X has {n_samples_X} samples, "
                f"confounds has {n_samples_confounds} samples"
            )
    
    return True


def check_data_quality(X: np.ndarray, Y: np.ndarray, 
                      confounds: Optional[np.ndarray] = None,
                      max_missing_rate: float = 0.1) -> Dict[str, Any]:
    """
    检查数据质量
    
    Args:
        X: 脑数据
        Y: 行为数据
        confounds: 混杂变量
        max_missing_rate: 最大允许缺失率
        
    Returns:
        数据质量报告
    """
    report = {
        'timestamp': create_timestamp(),
        'n_samples': X.shape[0],
        'n_features_X': X.shape[1],
        'n_features_Y': Y.shape[1]
    }
    
    # 检查缺失值
    X_missing_rate = np.isnan(X).sum() / X.size
    Y_missing_rate = np.isnan(Y).sum() / Y.size
    
    report['missing_rate_X'] = X_missing_rate
    report['missing_rate_Y'] = Y_missing_rate
    
    if confounds is not None:
        report['n_features_confounds'] = confounds.shape[1]
        confounds_missing_rate = np.isnan(confounds).sum() / confounds.size
        report['missing_rate_confounds'] = confounds_missing_rate
    
    # 检查异常值
    X_std = np.nanstd(X, axis=0)
    Y_std = np.nanstd(Y, axis=0)
    
    report['zero_variance_features_X'] = np.sum(X_std == 0)
    report['zero_variance_features_Y'] = np.sum(Y_std == 0)
    
    # 质量评估
    report['quality_passed'] = (
        X_missing_rate <= max_missing_rate and
        Y_missing_rate <= max_missing_rate and
        report['zero_variance_features_X'] == 0 and
        report['zero_variance_features_Y'] == 0
    )
    
    if confounds is not None:
        report['quality_passed'] = report['quality_passed'] and confounds_missing_rate <= max_missing_rate
    
    return report


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        
        if config_file is not None:
            self.load_config(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            'data': {
                'root_dir': str(get_data_dir()),
                'brain_file': 'fc_vector/Schaefer400/EFNY_Schaefer400_FC_matrix.npy',
                'behavioral_file': 'table/demo/EFNY_behavioral_data.csv',
                'sublist_file': 'table/sublist/sublist.txt'
            },
            'model': {
                'default_n_components': 5,
                'supported_models': ['pls', 'scca'],
                'default_model': 'pls'
            },
            'preprocessing': {
                'standardize_confounds': True,
                'max_missing_rate': 0.1
            },
            'evaluation': {
                'cv_n_splits': 5,
                'cv_shuffle': True,
                'permutation_n_iters': 1000
            },
            'output': {
                'results_dir': str(get_results_dir()),
                'save_formats': ['json', 'npz'],
                'create_timestamp_subdirs': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            logger = logging.getLogger(__name__)
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 递归更新配置
            self._update_config_recursive(self.config, user_config)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Configuration loaded from: {config_file}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading config file: {e}, using defaults")
    
    def _update_config_recursive(self, base_config: Dict, update_config: Dict) -> None:
        """递归更新配置"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 点分隔的键路径 (例如 'model.default_n_components')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_config(self, output_file: Union[str, Path]) -> None:
        """
        保存当前配置到文件
        
        Args:
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration saved to: {output_file}")


# 全局配置管理器实例
config = ConfigManager()