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
    
    # 回归混杂变量
    if args.regress_confounds:
        logger.info("Regressing out confounds")
        
        # NaN 插补（避免回归/PLS 直接失败）
        X_arr = brain_data.values if isinstance(brain_data, pd.DataFrame) else brain_data
        Y_arr = behavioral_data.values if isinstance(behavioral_data, pd.DataFrame) else behavioral_data
        C_arr = covariates.values if isinstance(covariates, pd.DataFrame) else covariates

        X_arr = _impute_with_mean(X_arr)
        Y_arr = _impute_with_mean(Y_arr)
        C_arr = _impute_with_mean(C_arr)

        # 创建混杂变量回归器（X/Y 分开，避免覆盖）
        confound_regressor_X = ConfoundRegressor(standardize=True)
        confound_regressor_Y = ConfoundRegressor(standardize=True)

        brain_clean = confound_regressor_X.fit_transform(X_arr, confounds=C_arr)
        behavioral_clean = confound_regressor_Y.fit_transform(Y_arr, confounds=C_arr)
        
        logger.info("Confound regression completed")
    else:
        brain_clean = brain_data
        behavioral_clean = behavioral_data
        logger.info("Skipping confound regression")
    
    return brain_clean, behavioral_clean


def create_model_instance(args):
    """创建模型实例"""
    logger = logging.getLogger(__name__)
    
    if args.model_type == 'adaptive_pls':
        # 自适应PLS模型 - 自动选择n_components
        model_params = {
            'n_components_range': list(range(1, args.n_components + 1)),  # 搜索范围
            'cv_folds': 5,
            'criterion': 'canonical_correlation',
            'scale': True,
            'max_iter': 5000,
            'tol': 1e-06
        }
        logger.info(f"Creating Adaptive-PLS model with component search range: 1-{args.n_components}")
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

        # 做法B：如果使用adaptive_pls，则在置换检验中固定成分数量，
        # 使用真实数据分析选出的最佳成分数，改用标准PLS模型
        if args.model_type == 'adaptive_pls':
            try:
                best_k = load_best_n_components(args.model_type)
                logger.info(f"Using fixed n_components={best_k} for permutation test (approach B)")
                from src.models.models import create_model
                base_model = create_model(
                    'pls',
                    n_components=best_k,
                    scale=True,
                    max_iter=5000,
                    tol=1e-06
                )
            except Exception as e:
                logger.error(
                    f"Failed to load best_n_components for permutation test, "
                    f"falling back to current model. Error: {e}"
                )
                base_model = model
        else:
            base_model = model
        
        # 运行置换检验
        perm_tester = PermutationTester(n_permutations=1, random_state=permutation_seed)
        result = perm_tester.run_permutation_test(
            base_model, brain_data, behavioral_data, confounds=covariates,
            permutation_seed=permutation_seed
        )
        
        result['task_type'] = 'permutation'
        result['task_id'] = args.task_id
        result['permutation_seed'] = permutation_seed
        
        logger.info(f"Permutation test completed. Max canonical correlation: {np.max(result['canonical_correlations']):.4f}")
        
    else:
        # 真实数据分析
        logger.info("Running real data analysis")
        
        # 拟合模型
        model.fit(brain_data, behavioral_data)
        
        # 获取结果
        X_scores, Y_scores = model.transform(brain_data, behavioral_data)
        canonical_corrs = model.calculate_canonical_correlations(X_scores, Y_scores)
        variance_explained = model.calculate_variance_explained(brain_data, behavioral_data, X_scores, Y_scores)
        X_loadings, Y_loadings = model.get_loadings()
        
        result = {
            'task_type': 'real_data',
            'task_id': 0,
            'model_info': model.get_model_info(),
            'canonical_correlations': canonical_corrs,
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
        
        # 运行交叉验证（可选）
        if args.run_cv:
            logger.info("Running cross-validation")
            cv_evaluator = CrossValidator(
                n_splits=args.cv_n_splits,
                shuffle=args.cv_shuffle,
                random_state=args.random_state
            )
            
            cv_results = cv_evaluator.run_cv_evaluation(model, brain_data, behavioral_data, confounds=covariates)
            result['cv_results'] = cv_results
            if 'all_canonical_correlations' in cv_results:
                result['cv_all_canonical_correlations'] = cv_results['all_canonical_correlations']
            
            logger.info("Cross-validation completed")
        
        logger.info(f"Real data analysis completed. Max canonical correlation: {np.max(canonical_corrs):.4f}")
    
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
        output_dir = (
            results_root
            / analysis_type
            / atlas_tag
            / args.model_type
            / f"ncomp_{args.n_components}"
            / f"seed_{seed}"
            / f"task_{args.task_id}"
        )
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建时间戳
    timestamp = create_timestamp()

    # 每次运行独立子目录，便于对比
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出文件名
    output_path = output_dir / "result"
    
    meta = result.get('metadata')
    if not isinstance(meta, dict):
        meta = {}

    # 添加元数据
    meta.update({
        'timestamp': timestamp,
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

    result['metadata'] = meta
    
    # 保存结果
    saved_files = save_results(result, output_path, format="both")

    logger.info(f"Results saved to:")
    for format_type, file_path in saved_files.items():
        logger.info(f"  {format_type}: {file_path}")

    # 如果是真实数据且使用自适应PLS，单独保存最佳成分数量，供置换检验使用
    if result.get('task_type') == 'real_data':
        model_info = result.get('model_info', {})
        if model_info.get('model_type') == 'Adaptive-PLS':
            optimal_n_components = model_info.get('optimal_n_components')
            if optimal_n_components is not None:
                summary_dir = results_root / "summary" / atlas_tag
                summary_dir.mkdir(parents=True, exist_ok=True)
                summary_path = summary_dir / f"best_n_components_{args.model_type}.json"
                summary_data = {
                    'model_type': args.model_type,
                    'optimal_n_components': int(optimal_n_components),
                    'timestamp': timestamp,
                    'source_result_json': str(saved_files.get('json', '')),
                    'source_result_npz': str(saved_files.get('npz', ''))
                }
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Best n_components summary saved to: {summary_path}")

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
