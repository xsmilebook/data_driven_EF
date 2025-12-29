#!/usr/bin/env python3
"""
主程序入口 - 支持真实数据分析和单个置换任务
支持 HPC 集群并行化
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import pandas as pd
import logging
import json

def _ensure_repo_root_on_syspath() -> Path:
    """
    Ensure repository root is on sys.path.

    Preferred invocation is `python -m scripts.run_single_task` from repo root.
    This helper keeps `python scripts/run_single_task.py ...` working without
    scattering sys.path hacks throughout the codebase.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from src.models.data_loader import EFNYDataLoader, create_synthetic_data
from src.models.models import create_model, get_available_models
from src.models.evaluation import run_nested_cv_evaluation
from src.models.utils import (
    setup_logging, get_results_dir, create_timestamp,
    save_results, save_large_artifacts, validate_data_shapes, check_data_quality,
    config
)
from src.config_io import load_simple_yaml
from src.path_config import load_paths_config, resolve_dataset_roots


def _infer_atlas_tag(brain_file: str) -> str:
    s = "" if brain_file is None else str(brain_file)
    s_low = s.lower()
    if "schaefer400" in s_low:
        return "schaefer400"
    if "schaefer100" in s_low:
        return "schaefer100"
    return "unknown_atlas"


def _build_behavioral_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    exclude = {"subject_code", "file_name", "subid", "id"}
    selected_measures = config.get('behavioral.selected_measures', [])

    if isinstance(selected_measures, list) and len(selected_measures) > 0:
        candidate_cols = [c for c in selected_measures if c in df.columns and c not in exclude]
        missing = [c for c in selected_measures if c not in df.columns]
        if missing:
            logging.getLogger(__name__).warning(
                f"Some selected_measures are missing in behavioral table and will be ignored: {missing}"
            )
        if len(candidate_cols) == 0:
            raise ValueError(
                "No selected_measures found in behavioral table. "
                "Please update configs/datasets/<DATASET>.yaml: behavioral.selected_measures to match EFNY_beh_metrics.csv column names."
            )
    else:
        candidate_cols = [c for c in df.columns if c not in exclude]

    # 强制转数值（宽表中可能存在 object dtype）
    Y = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
    metric_cols = list(Y.columns)
    return Y, metric_cols


def _impute_with_mean(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    means = np.nanmean(out, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    mask = np.isnan(out)
    if mask.any():
        out[mask] = np.take(means, np.where(mask)[1])
    return out


def _apply_yaml_configs(args) -> dict:
    """
    Load configs/paths.yaml + configs/datasets/<DATASET>.yaml and apply overrides
    to the global ConfigManager.

    Returns a small dict of resolved paths for logging/debugging.
    """
    paths_cfg = load_paths_config(args.paths_config, repo_root=REPO_ROOT)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)

    dataset_cfg_path = (
        Path(args.dataset_config)
        if args.dataset_config is not None
        else (REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml")
    )
    dataset_cfg = load_simple_yaml(dataset_cfg_path)
    analysis_cfg_path = Path(args.analysis_config) if args.analysis_config else None
    analysis_cfg = {}
    if analysis_cfg_path is not None and analysis_cfg_path.exists():
        analysis_cfg = load_simple_yaml(analysis_cfg_path)
    files = dataset_cfg.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"Invalid dataset config: files must be a mapping ({dataset_cfg_path})")

    overrides = {
        "data": {
            # For modeling, inputs are expected under data/processed/<DATASET>/ by default.
            "root_dir": str(roots["processed_root"]),
            "raw_root": str(roots["raw_root"]),
            "interim_root": str(roots["interim_root"]),
            "processed_root": str(roots["processed_root"]),
            "brain_file": str(files.get("brain_file", "")),
            "behavioral_file": str(files.get("behavioral_file", "")),
            "sublist_file": str(files.get("sublist_file", "")),
            "covariates_file": str(files.get("covariates_file", "")),
        },
        "output": {
            "results_dir": str(roots["outputs_root"] / "results"),
        },
    }

    if "preprocessing" in dataset_cfg:
        overrides["preprocessing"] = dataset_cfg["preprocessing"]
    if "behavioral" in dataset_cfg:
        overrides["behavioral"] = dataset_cfg["behavioral"]
    if isinstance(analysis_cfg, dict) and analysis_cfg:
        if "preprocessing" in analysis_cfg:
            overrides.setdefault("preprocessing", {})
            overrides["preprocessing"].update(analysis_cfg["preprocessing"])
        if "evaluation" in analysis_cfg:
            overrides["evaluation"] = analysis_cfg["evaluation"]

    config.apply_overrides(overrides)

    missing = [k for k in ("brain_file", "behavioral_file", "sublist_file") if not overrides["data"].get(k)]
    if missing and not args.use_synthetic:
        raise ValueError(f"Missing required dataset file keys in {dataset_cfg_path}: {missing}")

    return {
        "repo_root": roots["repo_root"],
        "raw_root": roots["raw_root"],
        "interim_root": roots["interim_root"],
        "processed_root": roots["processed_root"],
        "outputs_root": roots["outputs_root"],
        "dataset_cfg_path": dataset_cfg_path,
        "analysis_cfg_path": analysis_cfg_path,
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EFNY Brain-Behavior Association Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 运行真实数据分析
    python run_single_task.py --task_id 0 --model_type adaptive_pls
    
    # 运行置换检验（task_id=1-1000 代表不同的置换种子）
    python run_single_task.py --task_id 1 --model_type adaptive_pls --use_synthetic
    
    # 运行调参版本的 SCCA / rCCA 分析
    python run_single_task.py --task_id 0 --model_type adaptive_scca
    python run_single_task.py --task_id 0 --model_type adaptive_rcca
    
    # 批量运行（在 HPC 上使用数组作业）
    for i in {1..100}; do
        sbatch --wrap "python run_single_task.py --task_id $i --model_type adaptive_pls"
    done
        """
    )

    # Standardized entry-point args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (must exist in configs/paths.yaml and configs/datasets/<DATASET>.yaml).",
    )
    parser.add_argument(
        "--config",
        dest="paths_config",
        type=str,
        default="configs/paths.yaml",
        help="Centralized paths config (YAML).",
    )
    parser.add_argument(
        "--dataset-config",
        dest="dataset_config",
        type=str,
        default=None,
        help="Optional dataset config override (defaults to configs/datasets/<DATASET>.yaml).",
    )
    parser.add_argument(
        "--analysis-config",
        dest="analysis_config",
        type=str,
        default="configs/analysis.yaml",
        help="Optional analysis defaults config (YAML).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configs and validate imports without reading data or writing outputs.",
    )
    
    # 核心参数
    parser.add_argument(
        "--task_id", type=int, default=0,
        help="任务ID。0=真实数据分析；1-N=置换检验（使用不同的随机种子）"
    )
    
    parser.add_argument(
        "--model_type", type=str, default="adaptive_pls", choices=get_available_models(),
        help="模型类型（仅支持调参版本）：adaptive_pls / adaptive_scca / adaptive_rcca"
    )

    # 数据参数
    parser.add_argument(
        "--use_synthetic", action="store_true",
        help="使用合成数据（用于测试）"
    )
    
    parser.add_argument(
        "--covariates_path", type=str, default=None,
        help="协变量文件路径（.csv文件，可选）"
    )
    
    parser.add_argument(
        "--n_subjects", type=int, default=200,
        help="合成数据的被试数量"
    )
    
    parser.add_argument(
        "--atlas", type=str, default="schaefer100",
        help="合成数据使用的脑图谱名称（仅在use_synthetic时生效）"
    )
    
    parser.add_argument(
        "--n_brain_features", type=int, default=4950,
        help="合成数据的脑特征数量"
    )
    
    parser.add_argument(
        "--n_behavioral_measures", type=int, default=30,
        help="合成数据的行为指标数量"
    )
    
    # 预处理参数
    parser.add_argument(
        "--max_missing_rate", type=float, default=None,
        help="最大允许缺失率"
    )

    parser.add_argument(
        "--standardize_domains", dest="standardize_domains", action="store_true", default=None,
        help="? X/Y ??????"
    )
    parser.add_argument(
        "--no-standardize-domains", dest="standardize_domains", action="store_false", default=None,
        help="?? X/Y ??????"
    )

    parser.add_argument(
        "--pca_components_X", type=int, default=None,
        help="X ? PCA ???????"
    )
    parser.add_argument(
        "--pca_components_Y", type=int, default=None,
        help="Y ? PCA ???????"
    )
    
    # 评估参数
    parser.add_argument(
        "--run_cv", action="store_true", default=True,
        help="是否运行交叉验证"
    )
    
    parser.add_argument(
        "--cv_n_splits", type=int, default=None,
        help="交叉验证折数"
    )
    
    parser.add_argument(
        "--cv_shuffle", dest="cv_shuffle", action="store_true", default=None,
        help="????????"
    )
    parser.add_argument(
        "--no-cv-shuffle", dest="cv_shuffle", action="store_false", default=None,
        help="???????"
    )

    parser.add_argument(
        "--inner_cv_splits", type=int, default=None,
        help="???????????????"
    )

    parser.add_argument(
        "--inner_shuffle", dest="inner_shuffle", action="store_true", default=None,
        help="??????????????"
    )
    parser.add_argument(
        "--no-inner-shuffle", dest="inner_shuffle", action="store_false", default=None,
        help="?????????????"
    )

    parser.add_argument(
        "--outer_shuffle_random_state", type=int, default=None,
        help="?????????????"
    )

    parser.add_argument(
        "--inner_shuffle_random_state", type=int, default=None,
        help="?????????????"
    )

    parser.add_argument(
        "--permutation_n_iters", type=int, default=None,
        help="?????????????"
    )

    parser.add_argument(
        "--score_metric", type=str, default=None,
        help="??????"
    )

    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认使用项目结果目录）"
    )
    
    parser.add_argument(
        "--output_prefix", type=str, default="efny_analysis",
        help="输出文件前缀"
    )
    
    parser.add_argument(
        "--save_formats", type=str, nargs="+", default=["json", "npz"],
        choices=["json", "npz"],
        help="保存格式"
    )
    
    # 日志参数
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="日志文件路径"
    )
    
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def extract_covariates_from_behavioral_data(behavioral_data, subject_ids, covariates_path=None, logger=None):
    """从行为数据中提取协变量 - 仅支持age, sex, meanFD（EFNY标准）"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # EFNY标准协变量列表（不区分大小写）
    standard_covariates = ['age', 'sex', 'meanFD']
    
    if covariates_path:
        # 从指定文件加载协变量
        logger.info(f"Loading covariates from: {covariates_path}")
        covariates = pd.read_csv(covariates_path, encoding='utf-8')
        
        # 确保被试数量匹配
        if len(covariates) != len(subject_ids):
            logger.warning(f"Covariates shape mismatch: {len(covariates)} vs {len(subject_ids)}")
            # 尝试按索引对齐
            covariates = covariates.iloc[:len(subject_ids)]
        
        # 检查是否包含标准协变量（不区分大小写）
        covariates_lower = {col.lower(): col for col in covariates.columns}
        available_covs = []
        for std_cov in standard_covariates:
            if std_cov.lower() in covariates_lower:
                available_covs.append(covariates_lower[std_cov.lower()])
        
        if available_covs:
            logger.info(f"Found covariates in external file: {available_covs}")
            return covariates[available_covs]
        else:
            logger.warning("No standard covariates found in external file, using all columns")
            return covariates
    else:
        raise ValueError(
            "Covariates not found in behavioral table. "
            "Please pass --covariates_path or set files.covariates_file in configs/datasets/<DATASET>.yaml."
        )


def load_data(args):
    """加载数据 - 使用默认EFNY数据路径"""
    logger = logging.getLogger(__name__)
    
    if args.use_synthetic:
        logger.info("Using synthetic data for testing")
        brain_data, behavioral_data, covariates = create_synthetic_data(
            n_subjects=args.n_subjects,
            n_behavioral_measures=args.n_behavioral_measures,
            n_brain_features=args.n_brain_features,
            atlas=args.atlas,
            random_state=args.random_state
        )
        subject_ids = np.arange(args.n_subjects)
        # 选择用于分析的行为指标（通过配置）
        selected_measures = config.get('behavioral.selected_measures', [])
        if isinstance(behavioral_data, pd.DataFrame) and selected_measures:
            existing = [m for m in selected_measures if m in behavioral_data.columns]
            missing = [m for m in selected_measures if m not in behavioral_data.columns]
            if missing:
                logger.warning(f"Behavioral measures missing in synthetic data and will be skipped: {missing}")
            if existing:
                behavioral_data = behavioral_data[existing]
            else:
                raise ValueError(
                    "No selected_measures found in synthetic behavioral data. "
                    "Please update configs/datasets/<DATASET>.yaml behavioral.selected_measures or disable selection (set to [])."
                )
    else:
        logger.info("Loading real EFNY data using config paths")

        data_root = config.get('data.root_dir')
        brain_file = config.get('data.brain_file')
        behavioral_file = config.get('data.behavioral_file')
        sublist_file = config.get('data.sublist_file')

        data_loader = EFNYDataLoader(
            data_root=data_root,
            brain_file=brain_file,
            behavioral_file=behavioral_file,
            sublist_file=sublist_file,
        )

        brain_data, behavioral_raw, subject_ids = data_loader.load_all_data()

        # 协变量优先顺序：命令行 --covariates_path > config.data.covariates_file
        covariates_file = args.covariates_path or config.get('data.covariates_file')
        covariate_columns = config.get('preprocessing.confound_variables', [])

        try:
            covariates = data_loader.load_covariates_for_subjects(
                subject_ids,
                covariates_file=covariates_file,
                covariate_columns=covariate_columns,
            )
        except Exception as e:
            logger.error(f"Failed to load covariates aligned to sublist: {e}")
            raise

        # 使用 EFNY_beh_metrics.csv 的所有指标（排除 id 列，强制转数值）
        if not isinstance(behavioral_raw, pd.DataFrame):
            raise ValueError("Behavioral data must be a pandas DataFrame")

        behavioral_data, metric_columns = _build_behavioral_matrix(behavioral_raw)
        selected_measures = config.get('behavioral.selected_measures', [])
        if isinstance(selected_measures, list) and len(selected_measures) > 0:
            logger.info(f"Using SELECTED behavioral metrics: n_metrics={len(metric_columns)}")
        else:
            logger.info(f"Using ALL behavioral metrics: n_metrics={len(metric_columns)}")
    
    logger.info(f"Data loaded successfully:")
    logger.info(f"  Brain data shape: {brain_data.shape}")
    logger.info(f"  Behavioral data shape: {behavioral_data.shape}")
    logger.info(f"  Covariates shape: {covariates.shape}")
    logger.info(f"  Subject IDs: {len(subject_ids)}")
    
    return brain_data, behavioral_data, covariates, subject_ids


def preprocess_data(brain_data, behavioral_data, covariates, args):
    """预处理数据"""
    logger = logging.getLogger(__name__)
    
    # 数据质量检查
    quality_report = check_data_quality(
        brain_data, behavioral_data, covariates, 
        max_missing_rate=args.max_missing_rate
    )
    
    logger.info("Data quality report:")
    for key, value in quality_report.items():
        logger.info(f"  {key}: {value}")
    
    if not quality_report['quality_passed']:
        logger.warning("Data quality check failed, but continuing with analysis")
    
    return brain_data, behavioral_data


def create_model_instance(args):
    """创建模型实例"""
    logger = logging.getLogger(__name__)
    
    if args.model_type == 'adaptive_pls':
        # 自适应PLS模型 - 自动选择n_components
        model_params = {
            'n_components_range': [5],
            'cv_folds': 5,
            'criterion': 'first_component_correlation',
            'random_state': args.random_state,
            'scale': True,
            'max_iter': 5000,
            'tol': 1e-06,
        }
        logger.info("Creating Adaptive-PLS model with fixed n_components_range=[5]")
    elif args.model_type == 'adaptive_scca':
        # 自适应SCCA模型 - 自动选择成分数和稀疏度参数
        model_params = {
            'n_components_range': [5],
            'sparsity_X_range': [0.001, 0.005, 0.01, 0.05],  # 脑数据稀疏度（特征多，用小值）
            'sparsity_Y_range': [0.1, 0.2, 0.3],  # 行为数据稀疏度（特征少，用大值）
            'cv_folds': 5,
            'criterion': 'first_component_correlation',
            'random_state': args.random_state,
            'max_iter': 10000,
            'tol': 1e-06,
        }
        logger.info(f"Creating Adaptive-SCCA model with n_components range: {model_params['n_components_range']}")
        logger.info(f"sparsity_X range: {model_params['sparsity_X_range']}")
        logger.info(f"sparsity_Y range: {model_params['sparsity_Y_range']}")
        logger.info(f"Total combinations: {len(model_params['n_components_range']) * len(model_params['sparsity_X_range']) * len(model_params['sparsity_Y_range'])}")
    elif args.model_type == 'adaptive_rcca':
        # 自适应rCCA模型 - 自动选择成分数和正则化参数
        model_params = {
            'n_components_range': [5],
            'c_X_range': [0.0, 0.1, 0.3, 0.5],
            'c_Y_range': [0.0, 0.1, 0.3, 0.5],
            'cv_folds': 5,
            'criterion': 'first_component_correlation',
            'random_state': args.random_state,
            'pca': True,
            'eps': 1e-06,
        }
        logger.info(f"Creating Adaptive-rCCA model with n_components range: {model_params['n_components_range']}")
        logger.info(f"c_X range: {model_params['c_X_range']}")
        logger.info(f"c_Y range: {model_params['c_Y_range']}")
        logger.info(f"Total combinations: {len(model_params['n_components_range']) * len(model_params['c_X_range']) * len(model_params['c_Y_range'])}")
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    
    model = create_model(args.model_type, **model_params)
    
    return model


def run_analysis(model, brain_data, behavioral_data, covariates, args):
    """运行分析"""
    logger = logging.getLogger(__name__)

    def _load_real_stepwise_summary(results_root: Path, atlas_tag: str, model_type: str):
        summary_dir = results_root / "summary" / atlas_tag / model_type
        scores_path = summary_dir / "real_loadings_scores_summary.npz"
        subject_scores_path = summary_dir / "real_subject_train_scores.npz"
        x_loadings_path = summary_dir / "real_x_loadings_summary.npy"
        y_loadings_path = summary_dir / "real_y_loadings_summary.npy"

        if not scores_path.exists() or not subject_scores_path.exists():
            raise FileNotFoundError(
                f"Real summary not found in {summary_dir}. "
                "Run summarize_real_loadings_scores.py first."
            )

        scores = np.load(scores_path)
        subject_scores = np.load(subject_scores_path)
        x_loadings = np.load(x_loadings_path) if x_loadings_path.exists() else None
        y_loadings = np.load(y_loadings_path) if y_loadings_path.exists() else None

        return {
            "scores_mean": scores.get("scores_mean"),
            "X_scores_mean": subject_scores.get("X_scores_mean"),
            "Y_scores_mean": subject_scores.get("Y_scores_mean"),
            "x_loadings": x_loadings,
            "y_loadings": y_loadings,
            "summary_dir": summary_dir,
        }

    def _build_nested_param_candidates(model_for_grid):
        params0 = model_for_grid.get_params()
        n_range = params0.get('n_components_range', []) or [params0.get('n_components', 1)]
        candidates = []
        if args.model_type == 'adaptive_pls':
            for n_comp in n_range:
                p = dict(params0)
                p['n_components'] = int(n_comp)
                candidates.append(p)
            return candidates or [params0]
        if args.model_type == 'adaptive_scca':
            for n_comp in n_range:
                for sx in params0.get('sparsity_X_range', []):
                    for sy in params0.get('sparsity_Y_range', []):
                        p = dict(params0)
                        p['n_components'] = int(n_comp)
                        p['sparsity_X'] = float(sx)
                        p['sparsity_Y'] = float(sy)
                        candidates.append(p)
            return candidates or [params0]
        if args.model_type == 'adaptive_rcca':
            for n_comp in n_range:
                for cx in params0.get('c_X_range', []):
                    for cy in params0.get('c_Y_range', []):
                        p = dict(params0)
                        p['n_components'] = int(n_comp)
                        p['c_X'] = float(cx)
                        p['c_Y'] = float(cy)
                        candidates.append(p)
            return candidates or [params0]
        return [params0]
    # 置换检验模式
    if args.task_id > 0:
        logger.info(f"Running permutation test (task_id={args.task_id})")
        permutation_seed = args.random_state + args.task_id

        if args.model_type in ('adaptive_pls', 'adaptive_scca', 'adaptive_rcca'):
            from src.models.models import create_model

            perm_n_components_range = [5]

            if args.model_type == 'adaptive_pls':
                model_params = {
                    'n_components_range': perm_n_components_range,
                    'cv_folds': 5,
                    'criterion': 'first_component_correlation',
                    'random_state': permutation_seed,
                    'scale': True,
                    'max_iter': 5000,
                    'tol': 1e-06,
                }
            elif args.model_type == 'adaptive_scca':
                model_params = {
                    'n_components_range': perm_n_components_range,
                    'sparsity_X_range': [0.001, 0.005, 0.01, 0.05],
                    'sparsity_Y_range': [0.1, 0.2, 0.3],
                    'cv_folds': 5,
                    'criterion': 'first_component_correlation',
                    'random_state': permutation_seed,
                    'max_iter': 10000,
                    'tol': 1e-06,
                }
            else:
                model_params = {
                    'n_components_range': perm_n_components_range,
                    'c_X_range': [0.0, 0.1, 0.3, 0.5],
                    'c_Y_range': [0.0, 0.1, 0.3, 0.5],
                    'cv_folds': 5,
                    'criterion': 'first_component_correlation',
                    'random_state': permutation_seed,
                    'pca': True,
                    'eps': 1e-06,
                }

            logger.info(
                f"Permutation adaptive search: model={args.model_type}, search_range={perm_n_components_range}"
            )
            base_model = create_model(args.model_type, **model_params)
        else:
            base_model = model

        X_raw = brain_data.values if isinstance(brain_data, pd.DataFrame) else brain_data
        Y_raw = behavioral_data.values if isinstance(behavioral_data, pd.DataFrame) else behavioral_data
        C_raw = covariates.values if isinstance(covariates, pd.DataFrame) else covariates

        np.random.seed(permutation_seed)
        permuted_indices = np.random.permutation(len(Y_raw))
        Y_perm = Y_raw[permuted_indices]

        atlas_tag = _infer_atlas_tag(config.get('data.brain_file', ''))
        results_dir = config.get('output.results_dir')
        if not results_dir:
            raise ValueError("Missing config value: output.results_dir")
        results_root = Path(results_dir)
        real_summary = _load_real_stepwise_summary(results_root, atlas_tag, args.model_type)
        real_scores_mean = real_summary.get('scores_mean')
        real_X_scores = real_summary.get('X_scores_mean')
        real_Y_scores = real_summary.get('Y_scores_mean')

        if real_scores_mean is None or real_X_scores is None or real_Y_scores is None:
            raise ValueError('Real stepwise summary is missing required scores.')
        if real_X_scores.shape[0] != X_raw.shape[0] or real_Y_scores.shape[0] != X_raw.shape[0]:
            raise ValueError(
                f"Real stepwise scores subject count mismatch: "
                f"real={real_X_scores.shape[0]} vs X={X_raw.shape[0]}"
            )

        n_components = int(np.asarray(real_scores_mean).size)
        perm_stepwise_scores = []
        perm_stepwise_results = []

        for k in range(1, n_components + 1):
            extra_confounds = None
            if k > 1:
                extra_confounds = np.hstack([
                    real_X_scores[:, :k-1],
                    real_Y_scores[:, :k-1],
                ])
            if C_raw is not None and extra_confounds is not None:
                confounds_k = np.hstack([C_raw, extra_confounds])
            elif extra_confounds is not None:
                confounds_k = extra_confounds
            else:
                confounds_k = C_raw

            param_candidates = _build_nested_param_candidates(base_model)
            for p in param_candidates:
                p['n_components'] = 1

            outer_seed = (
                args.outer_shuffle_random_state
                if args.outer_shuffle_random_state is not None
                else permutation_seed
            )
            inner_seed = (
                args.inner_shuffle_random_state
                if args.inner_shuffle_random_state is not None
                else permutation_seed
            )

            nested_results = run_nested_cv_evaluation(
                base_model,
                X_raw,
                Y_perm,
                confounds=confounds_k,
                outer_cv_splits=args.cv_n_splits,
                inner_cv_splits=args.inner_cv_splits,
                param_grid=param_candidates,
                outer_shuffle=args.cv_shuffle,
                inner_shuffle=args.inner_shuffle,
                outer_random_state=outer_seed,
                inner_random_state=inner_seed,
                standardize_domains=args.standardize_domains,
                pca_components_X=args.pca_components_X,
                pca_components_Y=args.pca_components_Y,
                score_metric=args.score_metric,
            )

            k_score = float(np.asarray(nested_results.get('outer_mean_canonical_correlations', [np.nan]))[0])
            perm_stepwise_scores.append(k_score)
            perm_stepwise_results.append(nested_results)

        result = {
            'permutation_seed': permutation_seed,
            'stepwise_scores': np.asarray(perm_stepwise_scores, dtype=float),
            'stepwise_results': perm_stepwise_results,
            'n_components': int(n_components),
            'n_samples': int(len(X_raw)),
            'cv_n_splits': int(args.cv_n_splits),
            'task_type': 'permutation',
            'task_id': args.task_id,
            'real_scores_mean': np.asarray(real_scores_mean, dtype=float),
        }

        logger.info(
            f"Permutation stepwise completed. Max score={np.nanmax(perm_stepwise_scores):.4f}"
        )
    else:
        # 真实数据分析
        logger.info("Running real data analysis")

        X_raw = brain_data.values if isinstance(brain_data, pd.DataFrame) else brain_data
        Y_raw = behavioral_data.values if isinstance(behavioral_data, pd.DataFrame) else behavioral_data
        C_raw = covariates.values if isinstance(covariates, pd.DataFrame) else covariates

        param_candidates = _build_nested_param_candidates(model)
        outer_seed = (
            args.outer_shuffle_random_state
            if args.outer_shuffle_random_state is not None
            else args.random_state
        )
        inner_seed = (
            args.inner_shuffle_random_state
            if args.inner_shuffle_random_state is not None
            else args.random_state
        )

        nested_results = run_nested_cv_evaluation(
            model,
            X_raw,
            Y_raw,
            confounds=C_raw,
            outer_cv_splits=args.cv_n_splits,
            inner_cv_splits=args.inner_cv_splits,
            param_grid=param_candidates,
            outer_shuffle=args.cv_shuffle,
            inner_shuffle=args.inner_shuffle,
            outer_random_state=outer_seed,
            inner_random_state=inner_seed,
            standardize_domains=args.standardize_domains,
            pca_components_X=args.pca_components_X,
            pca_components_Y=args.pca_components_Y,
            score_metric=args.score_metric,
        )

        cv_mean_corrs = np.asarray(nested_results.get('outer_mean_canonical_correlations', []), dtype=float)
        
        result = {
            'task_type': 'real_data',
            'task_id': 0,
            'model_info': model.get_model_info(),
            'canonical_correlations': cv_mean_corrs,
            'mean_canonical_correlations': cv_mean_corrs,
            'n_samples': brain_data.shape[0],
            'n_features_X': brain_data.shape[1],
            'n_features_Y': behavioral_data.shape[1]
        }

        result['cv_results'] = nested_results
        if 'outer_all_test_canonical_correlations' in nested_results:
            result['cv_all_canonical_correlations'] = nested_results['outer_all_test_canonical_correlations']

        logger.info(
            "Real data analysis completed. "
            f"Max CV mean canonical correlation={np.max(cv_mean_corrs):.4f}"
        )

    return result


def save_results_with_metadata(result, args):
    """保存结果和元数据"""
    logger = logging.getLogger(__name__)

    results_dir = config.get('output.results_dir')
    if not results_dir:
        raise ValueError("Missing config value: output.results_dir")
    results_root = Path(results_dir)
    brain_file = config.get('data.brain_file', '')
    atlas_tag = _infer_atlas_tag(brain_file)
    analysis_type = 'perm' if args.task_id > 0 else 'real'
    seed = (args.random_state + args.task_id) if args.task_id > 0 else args.random_state
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = (
            results_root
            / analysis_type
            / atlas_tag
            / args.model_type
            / f"seed_{seed}"
        )
    else:
        output_dir = Path(args.output_dir) / args.model_type / f"seed_{seed}"
        logger.info(f"Using user-specified output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出文件名
    output_path = output_dir / "result"
    
    # 获取或创建元数据
    metadata = result.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}
    
    # 添加运行元数据
    metadata.update({
        'task_id': args.task_id,
        'model_type': args.model_type,
        'use_synthetic': args.use_synthetic,
        'run_cv': args.run_cv,
        'cv_n_splits': args.cv_n_splits if args.run_cv else None,
        'random_state': args.random_state,
        'permutation_seed': seed if args.task_id > 0 else None,
        'permutation_n_iters': args.permutation_n_iters,
        'output_root': str(results_root),
        'output_dir': str(output_dir),
        'output_file': str(output_path),
        'data_root': config.get('data.root_dir'),
        'brain_file': config.get('data.brain_file'),
        'behavioral_file': config.get('data.behavioral_file'),
        'covariates_file': args.covariates_path or config.get('data.covariates_file'),
        'atlas_tag': atlas_tag,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'slurm_job_name': os.environ.get('SLURM_JOB_NAME'),
        'config_snapshot': config.config
    })

    cv_results = result.get('cv_results', {})
    if isinstance(cv_results, dict) and "score_metric" in cv_results:
        metadata['cv_score_metric'] = cv_results["score_metric"]
    
    result['metadata'] = metadata

    artifacts_dir = output_dir / "artifacts"
    artifact_map = save_large_artifacts(result, artifacts_dir)
    if artifact_map:
        result['artifacts'] = artifact_map
        metadata['artifacts_dir'] = str(artifacts_dir)
    
    # 保存结果
    saved_files = save_results(result, output_path, format="both")
    
    logger.info(f"Results saved to:")
    for format_type, file_path in saved_files.items():
        logger.info(f"  {format_type}: {file_path}")
    
    # 对于 Adaptive 模型的真实数据分析，保存最优超参数汇总
    if args.task_id == 0:
        model_info = result.get('model_info', {})
        cv_results = result.get('cv_results', {})
        summary_dir = results_root / "summary" / atlas_tag
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        if args.model_type == 'adaptive_pls':
            optimal_n_components = model_info.get('optimal_n_components')
            if optimal_n_components is None and isinstance(cv_results, dict):
                outer_folds = cv_results.get('outer_fold_results', [])
                n_list = []
                for fr in outer_folds:
                    p = fr.get('best_params', {})
                    n_val = p.get('n_components')
                    if n_val is None:
                        n_range = p.get('n_components_range')
                        if isinstance(n_range, list) and n_range:
                            n_val = n_range[0]
                    if n_val is not None:
                        n_list.append(int(n_val))
                if n_list:
                    optimal_n_components = int(max(set(n_list), key=n_list.count))
            if optimal_n_components is not None:
                summary_path = summary_dir / f"best_n_components_{args.model_type}.json"
                summary_data = {
                    'optimal_n_components': optimal_n_components,
                    'model_type': args.model_type,
                    'atlas': atlas_tag,
                    'timestamp': metadata.get('timestamp', create_timestamp())
                }
                save_results(summary_data, summary_path)
                logger.info(f"Saved best n_components summary to {summary_path}")
        
        elif args.model_type == 'adaptive_scca':
            optimal_n_components = model_info.get('optimal_n_components')
            optimal_sparsity_X = model_info.get('optimal_sparsity_X')
            optimal_sparsity_Y = model_info.get('optimal_sparsity_Y')
            if (optimal_n_components is None or optimal_sparsity_X is None or optimal_sparsity_Y is None) and isinstance(cv_results, dict):
                outer_folds = cv_results.get('outer_fold_results', [])
                combos = []
                for fr in outer_folds:
                    p = fr.get('best_params', {})
                    n_val = p.get('n_components')
                    sx_val = p.get('sparsity_X')
                    sy_val = p.get('sparsity_Y')
                    if n_val is None:
                        n_range = p.get('n_components_range')
                        if isinstance(n_range, list) and n_range:
                            n_val = n_range[0]
                    if sx_val is None:
                        sx_range = p.get('sparsity_X_range')
                        if isinstance(sx_range, list) and sx_range:
                            sx_val = sx_range[0]
                    if sy_val is None:
                        sy_range = p.get('sparsity_Y_range')
                        if isinstance(sy_range, list) and sy_range:
                            sy_val = sy_range[0]
                    if n_val is not None and sx_val is not None and sy_val is not None:
                        combos.append((int(n_val), float(sx_val), float(sy_val)))
                if combos:
                    best_combo = max(set(combos), key=combos.count)
                    optimal_n_components, optimal_sparsity_X, optimal_sparsity_Y = best_combo
            if optimal_n_components is not None and optimal_sparsity_X is not None and optimal_sparsity_Y is not None:
                summary_path = summary_dir / f"best_hyperparameters_{args.model_type}.json"
                summary_data = {
                    'optimal_n_components': optimal_n_components,
                    'optimal_sparsity_X': optimal_sparsity_X,
                    'optimal_sparsity_Y': optimal_sparsity_Y,
                    'model_type': args.model_type,
                    'atlas': atlas_tag,
                    'timestamp': metadata.get('timestamp', create_timestamp())
                }
                save_results(summary_data, summary_path)
                logger.info(f"Saved best hyperparameters summary to {summary_path}")
        
        elif args.model_type == 'adaptive_rcca':
            optimal_n_components = model_info.get('optimal_n_components')
            optimal_c_X = model_info.get('optimal_c_X')
            optimal_c_Y = model_info.get('optimal_c_Y')
            if (optimal_n_components is None or optimal_c_X is None or optimal_c_Y is None) and isinstance(cv_results, dict):
                outer_folds = cv_results.get('outer_fold_results', [])
                combos = []
                for fr in outer_folds:
                    p = fr.get('best_params', {})
                    n_val = p.get('n_components')
                    cx_val = p.get('c_X')
                    cy_val = p.get('c_Y')
                    if n_val is None:
                        n_range = p.get('n_components_range')
                        if isinstance(n_range, list) and n_range:
                            n_val = n_range[0]
                    if cx_val is None:
                        cx_range = p.get('c_X_range')
                        if isinstance(cx_range, list) and cx_range:
                            cx_val = cx_range[0]
                    if cy_val is None:
                        cy_range = p.get('c_Y_range')
                        if isinstance(cy_range, list) and cy_range:
                            cy_val = cy_range[0]
                    if n_val is not None and cx_val is not None and cy_val is not None:
                        combos.append((int(n_val), float(cx_val), float(cy_val)))
                if combos:
                    best_combo = max(set(combos), key=combos.count)
                    optimal_n_components, optimal_c_X, optimal_c_Y = best_combo
            if optimal_n_components is not None and optimal_c_X is not None and optimal_c_Y is not None:
                summary_path = summary_dir / f"best_hyperparameters_{args.model_type}.json"
                summary_data = {
                    'optimal_n_components': optimal_n_components,
                    'optimal_c_X': optimal_c_X,
                    'optimal_c_Y': optimal_c_Y,
                    'model_type': args.model_type,
                    'atlas': atlas_tag,
                    'timestamp': metadata.get('timestamp', create_timestamp())
                }
                save_results(summary_data, summary_path)
                logger.info(f"Saved best hyperparameters summary to {summary_path}")
    
    return saved_files


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    resolved = _apply_yaml_configs(args)

    # 设置日志（dry-run 不写文件）
    logger = setup_logging(
        log_level=args.log_level,
        log_file=None if args.dry_run else args.log_file,
    )

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Paths config: {args.paths_config}")
    logger.info(f"Dataset config: {resolved['dataset_cfg_path']}")
    if resolved.get("analysis_cfg_path") is not None:
        logger.info(f"Analysis config: {resolved['analysis_cfg_path']}")
    logger.info(f"Resolved raw_root: {resolved['raw_root']}")
    logger.info(f"Resolved interim_root: {resolved['interim_root']}")
    logger.info(f"Resolved processed_root: {resolved['processed_root']}")
    logger.info(f"Resolved outputs_root: {resolved['outputs_root']}")

    if args.dry_run:
        logger.info("Dry-run: configuration resolved and imports succeeded.")
        return 0

    if args.max_missing_rate is None:
        args.max_missing_rate = float(config.get("preprocessing.max_missing_rate", 0.1))
    if args.cv_n_splits is None:
        args.cv_n_splits = int(config.get("evaluation.cv_n_splits", 5))
    if args.cv_shuffle is None:
        args.cv_shuffle = bool(config.get("evaluation.cv_shuffle", True))
    if args.inner_cv_splits is None:
        args.inner_cv_splits = int(config.get("evaluation.inner_cv_splits", 3))
    if args.inner_shuffle is None:
        args.inner_shuffle = bool(config.get("evaluation.inner_shuffle", True))
    if args.outer_shuffle_random_state is None:
        args.outer_shuffle_random_state = config.get("evaluation.outer_shuffle_random_state")
    if args.inner_shuffle_random_state is None:
        args.inner_shuffle_random_state = config.get("evaluation.inner_shuffle_random_state")
    if args.permutation_n_iters is None:
        args.permutation_n_iters = int(config.get("evaluation.permutation_n_iters", 1000))
    if args.score_metric is None:
        args.score_metric = str(config.get("evaluation.score_metric", "mean_canonical_correlation"))
    if args.standardize_domains is None:
        args.standardize_domains = bool(config.get("preprocessing.standardize_domains", False))
    if args.pca_components_X is None:
        args.pca_components_X = config.get("preprocessing.pca_components_X")
    if args.pca_components_Y is None:
        args.pca_components_Y = config.get("preprocessing.pca_components_Y")
    
    logger.info("="*80)
    logger.info("EFNY Brain-Behavior Association Analysis Started")
    logger.info("="*80)
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Synthetic Data: {args.use_synthetic}")
    
    try:
        t0 = time.time()

        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        brain_data, behavioral_data, covariates, subject_ids = load_data(args)

        behavioral_metric_columns = (
            behavioral_data.columns.tolist()
            if isinstance(behavioral_data, pd.DataFrame)
            else None
        )
        covariate_columns = (
            covariates.columns.tolist()
            if isinstance(covariates, pd.DataFrame)
            else None
        )
        requested_behavioral_measures = config.get('behavioral.selected_measures', [])

        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        brain_clean, behavioral_clean = preprocess_data(brain_data, behavioral_data, covariates, args)

        # Validate shapes
        validate_data_shapes(brain_clean, behavioral_clean)

        # Step 3: Create model
        logger.info("Step 3: Creating model...")
        model = create_model_instance(args)

        # Step 4: Run analysis
        logger.info("Step 4: Running analysis...")
        result = run_analysis(model, brain_clean, behavioral_clean, covariates, args)

        result.setdefault('metadata', {})
        result['metadata']['runtime_seconds'] = float(time.time() - t0)
        result['metadata']['behavioral_metric_columns'] = behavioral_metric_columns
        result['metadata']['requested_behavioral_measures'] = requested_behavioral_measures
        result['metadata']['covariate_columns'] = covariate_columns

        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        save_results_with_metadata(result, args)

        logger.info("=" * 80)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
