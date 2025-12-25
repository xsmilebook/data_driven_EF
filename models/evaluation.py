#!/usr/bin/env python3
"""
评估模块 - 交叉验证和评估指标计算
提供通用的 CV 框架，不依赖具体模型类型
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from .preprocessing import ConfoundRegressor
from .models import BaseBrainBehaviorModel

logger = logging.getLogger(__name__)


def run_nested_cv_evaluation(model: BaseBrainBehaviorModel,
                           X: Union[np.ndarray, pd.DataFrame],
                           Y: Union[np.ndarray, pd.DataFrame],
                           confounds: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           outer_cv_splits: int = 5,
                           inner_cv_splits: int = 3,
                           param_grid: Optional[Union[Dict[str, list], list]] = None,
                           outer_shuffle: bool = True,
                           inner_shuffle: bool = True,
                           outer_random_state: Optional[int] = None,
                           inner_random_state: Optional[int] = None,
                           standardize_domains: bool = False,
                           pca_components_X: Optional[int] = None,
                           pca_components_Y: Optional[int] = None) -> Dict[str, Any]:
    """
    运行嵌套交叉验证评估
    
    Args:
        model: 脑-行为模型
        X: 脑数据
        Y: 行为数据
        confounds: 混杂变量
        outer_cv_splits: 外层CV折数
        inner_cv_splits: 内层CV折数
        
    Returns:
        嵌套CV结果
    """
    logger.info(f"Starting nested CV: outer={outer_cv_splits}, inner={inner_cv_splits}")

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    if isinstance(Y, pd.DataFrame):
        Y_values = Y.values
    else:
        Y_values = Y

    if isinstance(confounds, pd.DataFrame):
        confounds_values = confounds.values
    else:
        confounds_values = confounds

    if param_grid is None:
        param_candidates = [model.get_params()]
    else:
        if isinstance(param_grid, dict):
            param_candidates = list(ParameterGrid(param_grid))
        elif isinstance(param_grid, list):
            param_candidates = param_grid
        else:
            raise ValueError("param_grid must be a dict (ParameterGrid style), a list of dicts, or None")

    def _impute_with_train_mean(train_arr: np.ndarray, test_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        train_out = train_arr.copy()
        test_out = test_arr.copy()
        means = np.nanmean(train_out, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        train_mask = np.isnan(train_out)
        if train_mask.any():
            train_out[train_mask] = np.take(means, np.where(train_mask)[1])
        test_mask = np.isnan(test_out)
        if test_mask.any():
            test_out[test_mask] = np.take(means, np.where(test_mask)[1])
        return train_out, test_out

    def _prep_two_domain(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_apply: np.ndarray,
        Y_apply: np.ndarray,
        C_train: Optional[np.ndarray],
        C_apply: Optional[np.ndarray],
        rng_seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if C_train is not None:
            C_train, C_apply = _impute_with_train_mean(C_train, C_apply)

        X_train, X_apply = _impute_with_train_mean(X_train, X_apply)
        Y_train, Y_apply = _impute_with_train_mean(Y_train, Y_apply)

        if C_train is not None:
            reg_X = ConfoundRegressor(standardize=True)
            reg_Y = ConfoundRegressor(standardize=True)
            X_train = reg_X.fit_transform(X_train, confounds=C_train)
            Y_train = reg_Y.fit_transform(Y_train, confounds=C_train)
            X_apply = reg_X.transform(X_apply, confounds=C_apply)
            Y_apply = reg_Y.transform(Y_apply, confounds=C_apply)

        if standardize_domains:
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            Y_train = scaler_Y.fit_transform(Y_train)
            X_apply = scaler_X.transform(X_apply)
            Y_apply = scaler_Y.transform(Y_apply)

        if pca_components_X is not None:
            pca_X = PCA(n_components=pca_components_X, random_state=rng_seed)
            X_train = pca_X.fit_transform(X_train)
            X_apply = pca_X.transform(X_apply)

        if pca_components_Y is not None:
            pca_Y = PCA(n_components=pca_components_Y, random_state=rng_seed)
            Y_train = pca_Y.fit_transform(Y_train)
            Y_apply = pca_Y.transform(Y_apply)

        return X_train, Y_train, X_apply, Y_apply

    def _fit_and_score(
        params: Dict[str, Any],
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        return_scores: bool = False,
    ) -> Tuple[float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        model_fold = model.__class__(**params)
        model_fold.fit(X_train, Y_train)
        X_scores, Y_scores = model_fold.transform(X_test, Y_test)
        corrs = np.asarray(model_fold.calculate_canonical_correlations(X_scores, Y_scores), dtype=float)
        fold_score = float(np.nanmean(corrs)) if corrs.size else float('nan')
        if return_scores:
            return fold_score, corrs, X_scores, Y_scores
        return fold_score, corrs, None, None

    outer_kf = KFold(
        n_splits=outer_cv_splits,
        shuffle=outer_shuffle,
        random_state=outer_random_state,
    )

    score_metric = "mean_canonical_correlation"

    outer_fold_results = []
    for outer_fold_idx, (train_out_idx, test_out_idx) in enumerate(outer_kf.split(X_values)):
        X_train_out = X_values[train_out_idx]
        Y_train_out = Y_values[train_out_idx]
        X_test_out = X_values[test_out_idx]
        Y_test_out = Y_values[test_out_idx]

        if confounds_values is not None:
            C_train_out = confounds_values[train_out_idx]
            C_test_out = confounds_values[test_out_idx]
        else:
            C_train_out, C_test_out = None, None

        inner_kf = KFold(
            n_splits=inner_cv_splits,
            shuffle=inner_shuffle,
            random_state=inner_random_state,
        )

        theta_records = []
        best_theta = param_candidates[0]
        best_mean_score = -np.inf
        best_std_score = float('nan')

        for theta in param_candidates:
            inner_fold_scores = []

            for inner_fold_idx, (train_in_rel, val_in_rel) in enumerate(inner_kf.split(X_train_out)):
                train_in_idx = train_out_idx[train_in_rel]
                val_in_idx = train_out_idx[val_in_rel]

                X_train_in = X_values[train_in_idx]
                Y_train_in = Y_values[train_in_idx]
                X_val_in = X_values[val_in_idx]
                Y_val_in = Y_values[val_in_idx]

                if confounds_values is not None:
                    C_train_in = confounds_values[train_in_idx]
                    C_val_in = confounds_values[val_in_idx]
                else:
                    C_train_in, C_val_in = None, None

                rng_seed = None
                if inner_random_state is not None:
                    rng_seed = int(inner_random_state + outer_fold_idx * 1000 + inner_fold_idx)

                X_train_in_p, Y_train_in_p, X_val_in_p, Y_val_in_p = _prep_two_domain(
                    X_train_in,
                    Y_train_in,
                    X_val_in,
                    Y_val_in,
                    C_train_in,
                    C_val_in,
                    rng_seed,
                )

                fold_score, fold_corrs, _, _ = _fit_and_score(
                    theta,
                    X_train_in_p,
                    Y_train_in_p,
                    X_val_in_p,
                    Y_val_in_p,
                )
                inner_fold_scores.append(fold_score)

            mean_score = float(np.nanmean(np.asarray(inner_fold_scores, dtype=float)))
            std_score = float(np.nanstd(np.asarray(inner_fold_scores, dtype=float)))
            theta_records.append(
                {
                    'params': theta,
                    'mean_corr': mean_score,
                    'std_corr': std_score,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'score_metric': score_metric,
                    'inner_fold_scores': inner_fold_scores,
                }
            )

            if np.isfinite(mean_score) and mean_score > best_mean_score:
                best_mean_score = mean_score
                best_std_score = std_score
                best_theta = theta

        rng_seed_outer = outer_random_state
        if rng_seed_outer is not None:
            rng_seed_outer = int(rng_seed_outer + outer_fold_idx)

        X_train_out_p, Y_train_out_p, X_test_out_p, Y_test_out_p = _prep_two_domain(
            X_train_out,
            Y_train_out,
            X_test_out,
            Y_test_out,
            C_train_out,
            C_test_out,
            rng_seed_outer,
        )

        test_score, test_corrs, X_test_scores, Y_test_scores = _fit_and_score(
            best_theta,
            X_train_out_p,
            Y_train_out_p,
            X_test_out_p,
            Y_test_out_p,
            return_scores=True,
        )
        train_score, train_corrs, X_train_scores, Y_train_scores = _fit_and_score(
            best_theta,
            X_train_out_p,
            Y_train_out_p,
            X_train_out_p,
            Y_train_out_p,
            return_scores=True,
        )
        x_loadings, y_loadings = None, None
        try:
            model_ref = model.__class__(**best_theta)
            model_ref.fit(X_train_out_p, Y_train_out_p)
            x_loadings, y_loadings = model_ref.get_loadings()
        except Exception:
            x_loadings, y_loadings = None, None
        outer_fold_results.append(
            {
                'outer_fold': outer_fold_idx,
                'train_out_idx': train_out_idx,
                'test_out_idx': test_out_idx,
                'best_params': best_theta,
                'best_inner_mean_score': best_mean_score,
                'best_inner_std_score': best_std_score,
                'score_metric': score_metric,
                'inner_cv_table': theta_records,
                'test_mean_corr': test_score,
                'test_mean_score': test_score,
                'test_canonical_correlations': test_corrs,
                'test_scores_X': X_test_scores,
                'test_scores_Y': Y_test_scores,
                'train_mean_corr': train_score,
                'train_mean_score': train_score,
                'train_canonical_correlations': train_corrs,
                'train_scores_X': X_train_scores,
                'train_scores_Y': Y_train_scores,
                'x_loadings': x_loadings,
                'y_loadings': y_loadings,
            }
        )

    outer_scores = np.asarray([r.get('test_mean_score', np.nan) for r in outer_fold_results], dtype=float)

    max_n_components = 0
    for r in outer_fold_results:
        corrs = r.get('test_canonical_correlations')
        if corrs is None:
            continue
        try:
            max_n_components = max(max_n_components, int(np.asarray(corrs).size))
        except Exception:
            continue

    all_test_corrs = np.full((len(outer_fold_results), max_n_components), np.nan, dtype=float)
    for i, r in enumerate(outer_fold_results):
        corrs = r.get('test_canonical_correlations')
        if corrs is None:
            continue
        arr = np.asarray(corrs, dtype=float).reshape(-1)
        n = min(arr.size, max_n_components)
        if n > 0:
            all_test_corrs[i, :n] = arr[:n]

    with np.errstate(all='ignore'):
        outer_mean_canonical_correlations = np.nanmean(all_test_corrs, axis=0) if max_n_components else np.array([])
        outer_std_canonical_correlations = np.nanstd(all_test_corrs, axis=0) if max_n_components else np.array([])

    nested_results = {
        'n_outer_splits': int(outer_cv_splits),
        'n_inner_splits': int(inner_cv_splits),
        'outer_fold_results': outer_fold_results,
        'outer_mean_score': float(np.nanmean(outer_scores)) if outer_scores.size else float('nan'),
        'outer_std_score': float(np.nanstd(outer_scores)) if outer_scores.size else float('nan'),
        'outer_mean_canonical_correlations': outer_mean_canonical_correlations,
        'outer_std_canonical_correlations': outer_std_canonical_correlations,
        'outer_all_test_canonical_correlations': all_test_corrs,
        'score_metric': score_metric,
        'nested_evaluation_completed': True,
        'param_grid_size': int(len(param_candidates)),
        'standardize_domains': bool(standardize_domains),
        'pca_components_X': pca_components_X,
        'pca_components_Y': pca_components_Y,
    }

    return nested_results
