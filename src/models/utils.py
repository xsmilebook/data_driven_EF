#!/usr/bin/env python3
"""
工具模块 - 日志配置、路径管理和通用工具函数

Engineering-only notes:
- This module must not hard-code dataset roots or create filesystem artifacts at import time.
- All filesystem paths are supplied by script entry points via config files.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json
import numpy as np
import pandas as pd


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
    Return the dataset data root directory from config.

    Notes:
    - This function does not create directories.
    - Script entry points must populate `config['data']['root_dir']`.
    """
    root = config.get("data.root_dir", "")
    if not root:
        raise ValueError("Missing config value: data.root_dir")
    return Path(root)


def get_results_dir() -> Path:
    """
    Return the results root directory from config.

    Notes:
    - This function does not create directories.
    - Script entry points must populate `config['output']['results_dir']`.
    """
    root = config.get("output.results_dir", "")
    if not root:
        raise ValueError("Missing config value: output.results_dir")
    return Path(root)


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


def save_large_artifacts(results: Dict[str, Any],
                         output_dir: Union[str, Path],
                         keys_to_extract: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """
    Save large numpy arrays to separate .npy files and replace them with references.

    Args:
        results: Results dictionary to scan and mutate in place.
        output_dir: Directory to store extracted arrays.
        keys_to_extract: Set of keys to extract when value is a numpy array.

    Returns:
        Mapping of artifact keys to metadata (path, shape, dtype).
    """
    if keys_to_extract is None:
        keys_to_extract = {
            "train_scores_X",
            "train_scores_Y",
            "test_scores_X",
            "test_scores_Y",
            "X_scores",
            "Y_scores",
            "x_loadings",
            "y_loadings",
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Dict[str, Any]] = {}

    def _save_array(arr: np.ndarray, artifact_key: str) -> Dict[str, Any]:
        safe_name = artifact_key.replace("/", "_")
        file_path = output_dir / f"{safe_name}.npy"
        np.save(file_path, arr)
        return {
            "path": str(file_path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    def _walk(obj: Any, path_parts: list) -> None:
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key in keys_to_extract and isinstance(value, np.ndarray):
                    artifact_key = "/".join(path_parts + [key]) if path_parts else key
                    artifacts[artifact_key] = _save_array(value, artifact_key)
                    obj[key] = {"artifact_key": artifact_key}
                else:
                    _walk(value, path_parts + [key])
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, path_parts + [str(idx)])

    _walk(results, [])
    return artifacts


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
    检查数据质量 - 支持pandas DataFrame和numpy数组
    
    Args:
        X: 脑数据 (DataFrame或ndarray)
        Y: 行为数据 (DataFrame或ndarray)
        confounds: 混杂变量 (DataFrame或ndarray)
        max_missing_rate: 最大允许缺失率
        
    Returns:
        数据质量报告
    """
    report = {
        'timestamp': create_timestamp(),
        'n_samples': X.shape[0],
        'n_features_X': X.select_dtypes(include=[np.number]).shape[1] if isinstance(X, pd.DataFrame) else X.shape[1],
        'n_features_Y': Y.select_dtypes(include=[np.number]).shape[1] if isinstance(Y, pd.DataFrame) else Y.shape[1]
    }
    
    # 检查缺失值 - 支持DataFrame和ndarray
    def calculate_missing_rate(data):
        """计算缺失值比例"""
        if isinstance(data, pd.DataFrame):
            # 只选择数值列，避免字符串列导致的错误
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 0.0
            return numeric_data.isnull().sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
        else:  # numpy array
            return np.isnan(data).sum() / data.size
    
    X_missing_rate = calculate_missing_rate(X)
    Y_missing_rate = calculate_missing_rate(Y)
    
    report['missing_rate_X'] = X_missing_rate
    report['missing_rate_Y'] = Y_missing_rate
    
    if confounds is not None:
        report['n_features_confounds'] = confounds.shape[1]
        confounds_missing_rate = calculate_missing_rate(confounds)
        report['missing_rate_confounds'] = confounds_missing_rate
    
    # 检查异常值 - 支持DataFrame和ndarray
    def calculate_zero_variance_features(data):
        """计算零方差特征数量"""
        if isinstance(data, pd.DataFrame):
            # 只选择数值列，避免字符串列导致的错误
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 0
            std_values = numeric_data.std(axis=0)
            return (std_values == 0).sum()
        else:  # numpy array
            std_values = np.nanstd(data, axis=0)
            return np.sum(std_values == 0)
    
    report['zero_variance_features_X'] = calculate_zero_variance_features(X)
    report['zero_variance_features_Y'] = calculate_zero_variance_features(Y)
    
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
    
    def __init__(self) -> None:
        """初始化配置管理器"""
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            'data': {
                # Script entry points must supply dataset-specific roots and file paths.
                'root_dir': '',
                'brain_file': '',
                'behavioral_file': '',
                'sublist_file': '',
                'covariates_file': ''
            },
            'behavioral': {
                # Dataset-specific selections must live in configs/datasets/<DATASET>.yaml.
                'selected_measures': []
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
                # Script entry points must supply output roots.
                'results_dir': '',
                'save_formats': ['json', 'npz'],
                'create_timestamp_subdirs': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Merge a config dict into the current config (recursive)."""
        if not isinstance(overrides, dict):
            raise TypeError("Config overrides must be a dict")
        self._update_config_recursive(self.config, overrides)
    
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
