"""
EFNY Brain-Behavior Association Analysis Models Package

This package provides modular tools for analyzing brain-behavior associations
using Partial Least Squares (PLS) and Sparse Canonical Correlation Analysis (Sparse-CCA).
"""

from .data_loader import EFNYDataLoader, create_synthetic_data
from .preprocessing import ConfoundRegressor, DataQualityFilter, create_preprocessing_pipeline
from .models import BaseBrainBehaviorModel, PLSModel, SparseCCAModel, AdaptivePLSModel, create_model, get_available_models
from .evaluation import CrossValidator, PermutationTester, run_nested_cv_evaluation
from .utils import setup_logging, get_project_root, get_data_dir, get_results_dir, ConfigManager, save_results

__version__ = "1.0.0"
__author__ = "EFNY Analysis Team"

__all__ = [
    # Data loading
    "EFNYDataLoader",
    "create_synthetic_data",
    
    # Preprocessing
    "ConfoundRegressor",
    "DataQualityFilter",
    "create_preprocessing_pipeline",
    
    # Models
    "BaseBrainBehaviorModel",
    "PLSModel",
    "SparseCCAModel",
    "AdaptivePLSModel",
    "create_model",
    "get_available_models",
    
    # Evaluation
    "CrossValidator",
    "PermutationTester",
    "run_nested_cv_evaluation",
    
    # Utilities
    "setup_logging",
    "get_project_root",
    "get_data_dir",
    "get_results_dir",
    "ConfigManager",
    "save_results"
]