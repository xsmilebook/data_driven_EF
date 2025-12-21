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

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_loader import EFNYDataLoader, create_synthetic_data
from src.models.preprocessing import ConfoundRegressor, create_preprocessing_pipeline
from src.models.models import create_model, get_available_models
from src.models.evaluation import CrossValidator, PermutationTester
from src.models.utils import (
    setup_logging, get_results_dir, create_timestamp,
    save_results, validate_data_shapes, check_data_quality,
    ConfigManager, config, load_results
)


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
                "Please update config.json: behavioral.selected_measures to match EFNY_beh_metrics.csv column names."
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EFNY Brain-Behavior Association Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 运行真实数据分析
    python run_single_task.py --task_id 0 --model_type pls --n_components 5
    
    # 运行置换检验（task_id=1-1000 代表不同的置换种子）
    python run_single_task.py --task_id 1 --model_type pls --n_components 5 --use_synthetic
    
    # 运行 Sparse-CCA 分析
    python run_single_task.py --task_id 0 --model_type scca --n_components 3
    
    # 批量运行（在 HPC 上使用数组作业）
    for i in {1..100}; do
        sbatch --wrap "python run_single_task.py --task_id $i --model_type pls --n_perms 100"
    done
        """
    )
    
    # 核心参数
    parser.add_argument(
        "--task_id", type=int, default=0,
        help="任务ID。0=真实数据分析；1-N=置换检验（使用不同的随机种子）"
    )
    
    parser.add_argument(
        "--model_type", type=str, default="pls", choices=get_available_models(),
        help="模型类型：pls=偏最小二乘，scca=稀疏典型相关分析"
    )
    
    parser.add_argument(
        "--n_components", type=int, default=5,
        help="成分数量"
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
        "--regress_confounds", action="store_true", default=True,
        help="是否回归混杂变量（年龄、性别、头动）"
    )
    
    parser.add_argument(
        "--max_missing_rate", type=float, default=0.1,
        help="最大允许缺失率"
    )
    
    # 评估参数
    parser.add_argument(
        "--run_cv", action="store_true", default=True,
        help="是否运行交叉验证"
    )
    
    parser.add_argument(
        "--cv_n_splits", type=int, default=5,
        help="交叉验证折数"
    )
    
    parser.add_argument(
        "--cv_shuffle", action="store_true", default=True,
        help="是否打乱数据顺序"
    )
    
    # 输出参数
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
    
    # 配置参数
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="配置文件路径"
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
            "Please pass --covariates_path or set data.covariates_file in config.json."
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
                    "Please update config.json behavioral.selected_measures or disable selection (set to [])."
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
        covariate_columns = config.get('preprocessing.confound_variables', ['sex', 'age', 'meanFD'])

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
    
    if args.regress_confounds:
        logger.info(
            "Strict mode: confounds will be regressed within CV folds (train-fit/test-apply). "
            "Skipping global confound regression in preprocess_data."
        )
    else:
        logger.info("Confound regression disabled")

    return brain_data, behavioral_data


def create_model_instance(args):
    """创建模型实例"""
    logger = logging.getLogger(__name__)
    
    if args.model_type == 'adaptive_pls':
        # 自适应PLS模型 - 自动选择n_components
        model_params = {
            'n_components_range': list(range(1, args.n_components + 1)),  # 搜索范围
            'cv_folds': 5,
            'criterion': 'canonical_correlation',
            'random_state': args.random_state,
            'scale': True,
            'max_iter': 5000,
            'tol': 1e-06
        }
        logger.info(f"Creating Adaptive-PLS model with component search range: 1-{args.n_components}")
    elif args.model_type == 'adaptive_scca':
        # 自适应SCCA模型 - 自动选择成分数和稀疏度参数
        model_params = {
            'n_components_range': list(range(2, args.n_components + 1)),  # 搜索范围 [2, 3, ..., n_components]
            'sparsity_X_range': [0.001, 0.005, 0.01, 0.05],  # 脑数据稀疏度（特征多，用小值）
            'sparsity_Y_range': [0.1, 0.2, 0.3],  # 行为数据稀疏度（特征少，用大值）
            'cv_folds': 5,
            'criterion': 'canonical_correlation',
            'random_state': args.random_state,
            'max_iter': 10000,
            'tol': 1e-06
        }
        logger.info(f"Creating Adaptive-SCCA model with n_components range: {model_params['n_components_range']}")
        logger.info(f"sparsity_X range: {model_params['sparsity_X_range']}")
        logger.info(f"sparsity_Y range: {model_params['sparsity_Y_range']}")
        logger.info(f"Total combinations: {len(model_params['n_components_range']) * len(model_params['sparsity_X_range']) * len(model_params['sparsity_Y_range'])}")
    elif args.model_type == 'adaptive_rcca':
        # 自适应rCCA模型 - 自动选择成分数和正则化参数
        model_params = {
            'n_components_range': list(range(2, args.n_components + 1)),  # 搜索范围 [2, 3, ..., n_components]
            'c_X_range': [0.0, 0.1, 0.3, 0.5],
            'c_Y_range': [0.0, 0.1, 0.3, 0.5],
            'cv_folds': 5,
            'criterion': 'canonical_correlation',
            'random_state': args.random_state,
            'pca': True,
            'eps': 1e-06
        }
        logger.info(f"Creating Adaptive-rCCA model with n_components range: {model_params['n_components_range']}")
        logger.info(f"c_X range: {model_params['c_X_range']}")
        logger.info(f"c_Y range: {model_params['c_Y_range']}")
        logger.info(f"Total combinations: {len(model_params['n_components_range']) * len(model_params['c_X_range']) * len(model_params['c_Y_range'])}")
    else:
        # 标准模型
        model_params = {
            'n_components': args.n_components
        }
        
        if args.model_type == 'pls':
            model_params.update({
                'scale': True,
                'max_iter': 5000,
                'tol': 1e-06
            })
        elif args.model_type == 'scca':
            model_params.update({
                'sparsity_X': 0.1,
                'sparsity_Y': 0.1,
                'max_iter': 10000,
                'tol': 1e-06
            })
        elif args.model_type == 'rcca':
            model_params.update({
                'c_X': 0.1,
                'c_Y': 0.1,
                'pca': True,
                'eps': 1e-06
            })
        
        logger.info(f"Created {args.model_type.upper()} model with {args.n_components} components")
    
    model = create_model(args.model_type, **model_params)
    
    return model


def run_analysis(model, brain_data, behavioral_data, covariates, args):
    """运行分析"""
    logger = logging.getLogger(__name__)
    
    # 置换检验模式
    if args.task_id > 0:
        logger.info(f"Running permutation test (task_id={args.task_id})")
        
        # 设置置换种子
        permutation_seed = args.random_state + args.task_id

        if args.model_type in ('adaptive_pls', 'adaptive_scca', 'adaptive_rcca'):
            from src.models.models import create_model

            min_n_components = load_max_optimal_n_components_from_real_runs(args.model_type)
            upper_n_components = max(int(args.n_components), int(min_n_components))
            perm_n_components_range = list(range(int(min_n_components), int(upper_n_components) + 1))

            if args.model_type == 'adaptive_pls':
                model_params = {
                    'n_components_range': perm_n_components_range,
                    'cv_folds': 5,
                    'criterion': 'canonical_correlation',
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
                    'criterion': 'canonical_correlation',
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
                    'criterion': 'canonical_correlation',
                    'random_state': permutation_seed,
                    'pca': True,
                    'eps': 1e-06,
                }

            logger.info(
                f"Permutation adaptive search: model={args.model_type}, "
                f"min_n_components(from real max)={min_n_components}, "
                f"search_range={perm_n_components_range}"
            )
            base_model = create_model(args.model_type, **model_params)
        else:
            base_model = model
        
        # 运行置换检验
        perm_tester = PermutationTester(n_permutations=1, random_state=permutation_seed)
        result = perm_tester.run_permutation_test(
            base_model, brain_data, behavioral_data, confounds=covariates,
            permutation_seed=permutation_seed,
            cv_n_splits=args.cv_n_splits,
            cv_shuffle=args.cv_shuffle,
            cv_random_state=permutation_seed,
            return_cv_results=True,
        )
        
        result['task_type'] = 'permutation'
        result['task_id'] = args.task_id
        result['permutation_seed'] = permutation_seed
        
        logger.info(
            "Permutation test completed. "
            f"vector3(stat)={result.get('cv_statistic_vector3', float('nan')):.6f}; "
            f"Max CV mean canonical correlation={np.max(result['canonical_correlations']):.4f}"
        )
        
    else:
        # 真实数据分析
        logger.info("Running real data analysis")

        X_full = brain_data.values if isinstance(brain_data, pd.DataFrame) else brain_data
        Y_full = behavioral_data.values if isinstance(behavioral_data, pd.DataFrame) else behavioral_data
        C_full = covariates.values if isinstance(covariates, pd.DataFrame) else covariates

        X_full = _impute_with_mean(X_full)
        Y_full = _impute_with_mean(Y_full)
        if C_full is not None:
            C_full = _impute_with_mean(C_full)

        if args.regress_confounds and C_full is not None:
            confound_regressor_X = ConfoundRegressor(standardize=True)
            confound_regressor_Y = ConfoundRegressor(standardize=True)
            X_fit = confound_regressor_X.fit_transform(X_full, confounds=C_full)
            Y_fit = confound_regressor_Y.fit_transform(Y_full, confounds=C_full)
        else:
            X_fit, Y_fit = X_full, Y_full

        model.fit(X_fit, Y_fit)
        X_scores, Y_scores = model.transform(X_fit, Y_fit)
        in_sample_canonical_corrs = model.calculate_canonical_correlations(X_scores, Y_scores)
        variance_explained = model.calculate_variance_explained(X_fit, Y_fit, X_scores, Y_scores)
        X_loadings, Y_loadings = model.get_loadings()

        cv_evaluator = CrossValidator(
            n_splits=args.cv_n_splits,
            shuffle=args.cv_shuffle,
            random_state=args.random_state,
        )
        cv_results = cv_evaluator.run_cv_evaluation(model, brain_data, behavioral_data, confounds=covariates)
        cv_mean_corrs = np.asarray(cv_results.get('mean_canonical_correlations', []), dtype=float)
        cv_vector3_stat = float(np.linalg.norm(cv_mean_corrs)) if cv_mean_corrs.size else float('nan')
        
        result = {
            'task_type': 'real_data',
            'task_id': 0,
            'model_info': model.get_model_info(),
            'canonical_correlations': cv_mean_corrs,
            'mean_canonical_correlations': cv_mean_corrs,
            'cv_statistic_vector3': cv_vector3_stat,
            'in_sample_canonical_correlations': in_sample_canonical_corrs,
            'variance_explained_X': variance_explained['variance_explained_X'],
            'variance_explained_Y': variance_explained['variance_explained_Y'],
            'X_scores': X_scores,
            'Y_scores': Y_scores,
            'X_loadings': X_loadings,
            'Y_loadings': Y_loadings,
            'n_samples': brain_data.shape[0],
            'n_features_X': brain_data.shape[1],
            'n_features_Y': behavioral_data.shape[1]
        }

        # 严格A版总是保存 CV 结果（用于复核/复用）
        result['cv_results'] = cv_results
        if 'all_canonical_correlations' in cv_results:
            result['cv_all_canonical_correlations'] = cv_results['all_canonical_correlations']

        logger.info(
            "Real data analysis completed. "
            f"vector3(stat)={cv_vector3_stat:.6f}; "
            f"Max CV mean canonical correlation={np.max(cv_mean_corrs):.4f}"
        )

    return result


def save_results_with_metadata(result, args):
    """保存结果和元数据"""
    logger = logging.getLogger(__name__)

    results_root = Path(config.get('output.results_dir', str(get_results_dir())))
    brain_file = config.get('data.brain_file', '')
    atlas_tag = _infer_atlas_tag(brain_file)
    analysis_type = 'perm' if args.task_id > 0 else 'real'
    seed = (args.random_state + args.task_id) if args.task_id > 0 else args.random_state
    
    # 确定输出目录
    if args.output_dir is None:
        # 根据模型类型设置路径
        if args.model_type in ('adaptive_scca', 'adaptive_rcca'):
            # adaptive_* 的路径不包含 ncomp，因为实际超参数并不等于 args.n_components
            output_dir = (
                results_root
                / analysis_type
                / atlas_tag
                / args.model_type
                / f"seed_{seed}"
            )
        else:
            output_dir = (
                results_root
                / analysis_type
                / atlas_tag
                / args.model_type
                / f"ncomp_{args.n_components}"
                / f"seed_{seed}"
            )
    else:
        if args.model_type in ('adaptive_scca', 'adaptive_rcca'):
            output_dir = Path(args.output_dir) / args.model_type / f"seed_{seed}"
        else:
            output_dir = Path(args.output_dir) / args.model_type / f"ncomp_{args.n_components}" / f"seed_{seed}"
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
        'n_components': args.n_components,
        'use_synthetic': args.use_synthetic,
        'regress_confounds': args.regress_confounds,
        'run_cv': args.run_cv,
        'cv_n_splits': args.cv_n_splits if args.run_cv else None,
        'random_state': args.random_state,
        'permutation_seed': seed if args.task_id > 0 else None,
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
    
    result['metadata'] = metadata
    
    # 保存结果
    saved_files = save_results(result, output_path, format="both")
    
    logger.info(f"Results saved to:")
    for format_type, file_path in saved_files.items():
        logger.info(f"  {format_type}: {file_path}")
    
    # 对于 Adaptive 模型的真实数据分析，保存最优超参数汇总
    if args.task_id == 0:
        model_info = result.get('model_info', {})
        summary_dir = results_root / "summary" / atlas_tag
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        if args.model_type == 'adaptive_pls':
            optimal_n_components = model_info.get('optimal_n_components')
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


def load_best_n_components(model_type: str) -> int:
    """从汇总文件中加载真实数据选出的最佳成分数量"""
    results_root = Path(config.get('output.results_dir', str(get_results_dir())))
    brain_file = config.get('data.brain_file', '')
    atlas_tag = _infer_atlas_tag(brain_file)
    summary_path = results_root / "summary" / atlas_tag / f"best_n_components_{model_type}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Best n_components summary not found: {summary_path}. "
            f"请先运行 task_id=0 的真实数据分析以生成该文件。"
        )
    summary = load_results(summary_path)
    optimal_n_components = summary.get('optimal_n_components')
    if optimal_n_components is None:
        raise ValueError(
            f"No 'optimal_n_components' found in summary file: {summary_path}"
        )
    return int(optimal_n_components)


def load_best_hyperparameters(model_type: str) -> dict:
    """从汇总文件中加载真实数据选出的最佳超参数"""
    results_root = Path(config.get('output.results_dir', str(get_results_dir())))
    brain_file = config.get('data.brain_file', '')
    atlas_tag = _infer_atlas_tag(brain_file)
    summary_path = results_root / "summary" / atlas_tag / f"best_hyperparameters_{model_type}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Best hyperparameters summary not found: {summary_path}. "
            f"请先运行 task_id=0 的真实数据分析以生成该文件。"
        )
    summary = load_results(summary_path)
    if model_type == 'adaptive_scca':
        n_components = summary.get('optimal_n_components')
        sparsity_X = summary.get('optimal_sparsity_X')
        sparsity_Y = summary.get('optimal_sparsity_Y')
        if n_components is None or sparsity_X is None or sparsity_Y is None:
            raise ValueError(
                f"No optimal hyperparameters found in summary file: {summary_path}"
            )
        return {
            'n_components': int(n_components),
            'sparsity_X': float(sparsity_X), 
            'sparsity_Y': float(sparsity_Y)
        }
    elif model_type == 'adaptive_rcca':
        n_components = summary.get('optimal_n_components')
        c_X = summary.get('optimal_c_X')
        c_Y = summary.get('optimal_c_Y')
        if n_components is None or c_X is None or c_Y is None:
            raise ValueError(
                f"No optimal hyperparameters found in summary file: {summary_path}"
            )
        return {
            'n_components': int(n_components),
            'c_X': float(c_X),
            'c_Y': float(c_Y)
        }
    else:
        raise ValueError(f"Unsupported model type for hyperparameter loading: {model_type}")


def load_max_optimal_n_components_from_real_runs(model_type: str) -> int:
    """扫描 real 结果目录，提取多次 real run 的 optimal_n_components 最大值。

    该函数用于置换检验（严格版）：置换时 n_components 搜索下限不得低于 real 的最大最优成分数。
    """
    results_root = Path(config.get('output.results_dir', str(get_results_dir())))
    brain_file = config.get('data.brain_file', '')
    atlas_tag = _infer_atlas_tag(brain_file)
    real_root = results_root / 'real' / atlas_tag / model_type
    if not real_root.exists():
        raise FileNotFoundError(
            f"Real results directory not found for model={model_type}: {real_root}. "
            "请先运行多次 task_id=0 的真实数据分析以生成 seed_* 目录。"
        )

    def _find_result_file(seed_dir: Path) -> Path | None:
        json_path = seed_dir / 'result.json'
        if json_path.exists():
            return json_path
        npz_path = seed_dir / 'result.npz'
        if npz_path.exists():
            return npz_path
        return None

    values: list[int] = []
    for seed_dir in sorted(real_root.glob('seed_*')):
        res_path = _find_result_file(seed_dir)
        if res_path is None:
            continue
        try:
            res = load_results(res_path)
        except Exception:
            continue
        model_info = res.get('model_info', {}) if isinstance(res, dict) else {}
        opt = None
        if isinstance(model_info, dict):
            opt = model_info.get('optimal_n_components')
        if opt is None:
            continue
        try:
            values.append(int(opt))
        except Exception:
            continue

    if not values:
        raise RuntimeError(
            f"No valid optimal_n_components found under {real_root}. "
            "Ensure real-data results contain model_info.optimal_n_components."
        )

    return int(max(values))


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    logger.info("="*80)
    logger.info("EFNY Brain-Behavior Association Analysis Started")
    logger.info("="*80)
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Components: {args.n_components}")
    logger.info(f"Synthetic Data: {args.use_synthetic}")
    
    try:
        t0 = time.time()
        # 加载配置（如果提供）
        if args.config_file:
            config.load_config(args.config_file)
            logger.info(f"Configuration loaded from: {args.config_file}")
        
        # 步骤1: 加载数据
        logger.info("Step 1: Loading data...")
        brain_data, behavioral_data, covariates, subject_ids = load_data(args)

        behavioral_metric_columns = behavioral_data.columns.tolist() if isinstance(behavioral_data, pd.DataFrame) else None
        covariate_columns = covariates.columns.tolist() if isinstance(covariates, pd.DataFrame) else None
        requested_behavioral_measures = config.get('behavioral.selected_measures', [])
        
        # 步骤2: 预处理
        logger.info("Step 2: Preprocessing data...")
        brain_clean, behavioral_clean = preprocess_data(brain_data, behavioral_data, covariates, args)
        
        # 验证数据
        validate_data_shapes(brain_clean, behavioral_clean)
        
        # 步骤3: 创建模型
        logger.info("Step 3: Creating model...")
        model = create_model_instance(args)
        
        # 步骤4: 运行分析
        logger.info("Step 4: Running analysis...")
        result = run_analysis(model, brain_clean, behavioral_clean, covariates, args)

        result.setdefault('metadata', {})
        result['metadata']['runtime_seconds'] = float(time.time() - t0)
        result['metadata']['behavioral_metric_columns'] = behavioral_metric_columns
        result['metadata']['requested_behavioral_measures'] = requested_behavioral_measures
        result['metadata']['covariate_columns'] = covariate_columns
        
        # 步骤5: 保存结果
        logger.info("Step 5: Saving results...")
        saved_files = save_results_with_metadata(result, args)
        
        logger.info("="*80)
        logger.info("Analysis completed successfully!")
        logger.info("="*80)
        
        # 返回成功状态码
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
