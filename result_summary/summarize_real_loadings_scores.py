import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import load_results


def _load_artifact(result: dict, artifact_key: str) -> np.ndarray | None:
    artifacts = result.get("artifacts", {})
    entry = artifacts.get(artifact_key)
    if not entry:
        return None
    path = entry.get("path")
    if not path:
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _iter_real_results(results_root: Path, atlas: str | None, model_type: str | None):
    base = results_root / "real"
    if not base.exists():
        return
    if atlas is not None and model_type is not None:
        base = base / atlas / model_type
        json_glob = "**/seed_*/result.json"
        npz_glob = "**/seed_*/result.npz"
    elif atlas is not None:
        base = base / atlas
        json_glob = "**/seed_*/result.json"
        npz_glob = "**/seed_*/result.npz"
    elif model_type is not None:
        json_glob = f"**/{model_type}/seed_*/result.json"
        npz_glob = f"**/{model_type}/seed_*/result.npz"
    else:
        json_glob = "**/seed_*/result.json"
        npz_glob = "**/seed_*/result.npz"

    for p in base.glob(json_glob):
        yield p
    for p in base.glob(npz_glob):
        yield p


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--atlas", type=str, default=None)
    ap.add_argument("--model_type", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_root / "summary"
        if args.atlas:
            output_dir = output_dir / args.atlas
        if args.model_type:
            output_dir = output_dir / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_scores = []
    all_fold_x_loadings = []
    all_fold_y_loadings = []

    run_subject_scores_x_train = []
    run_subject_scores_y_train = []
    run_subject_scores_x_test = []
    run_subject_scores_y_test = []

    for path in _iter_real_results(results_root, args.atlas, args.model_type):
        try:
            res = load_results(path)
        except Exception:
            continue

        outer_folds = res.get("cv_results", {}).get("outer_fold_results", [])
        if not outer_folds:
            continue

        max_index = -1
        for fold in outer_folds:
            train_idx = fold.get("train_out_idx", [])
            test_idx = fold.get("test_out_idx", [])
            if len(train_idx):
                max_index = max(max_index, int(np.max(train_idx)))
            if len(test_idx):
                max_index = max(max_index, int(np.max(test_idx)))
        if max_index < 0:
            continue
        n_subjects = max_index + 1

        x_scores_sum = None
        y_scores_sum = None
        x_test_sum = None
        y_test_sum = None
        score_counts = np.zeros((n_subjects, 1), dtype=int)
        test_counts = np.zeros((n_subjects, 1), dtype=int)

        for fold in outer_folds:
            corrs = fold.get("test_canonical_correlations")
            if corrs is not None:
                all_fold_scores.append(np.asarray(corrs, dtype=float).reshape(-1))

            x_art = fold.get("x_loadings")
            y_art = fold.get("y_loadings")
            if isinstance(x_art, dict) and "artifact_key" in x_art:
                x_arr = _load_artifact(res, x_art["artifact_key"])
            else:
                x_arr = np.asarray(x_art) if x_art is not None else None
            if isinstance(y_art, dict) and "artifact_key" in y_art:
                y_arr = _load_artifact(res, y_art["artifact_key"])
            else:
                y_arr = np.asarray(y_art) if y_art is not None else None
            if x_arr is not None and y_arr is not None:
                all_fold_x_loadings.append(x_arr)
                all_fold_y_loadings.append(y_arr)

            train_idx = fold.get("train_out_idx")
            if train_idx is None:
                continue
            train_idx = np.asarray(train_idx, dtype=int).reshape(-1)

            x_scores_ref = fold.get("train_scores_X")
            y_scores_ref = fold.get("train_scores_Y")
            if isinstance(x_scores_ref, dict) and "artifact_key" in x_scores_ref:
                x_scores = _load_artifact(res, x_scores_ref["artifact_key"])
            else:
                x_scores = np.asarray(x_scores_ref) if x_scores_ref is not None else None
            if isinstance(y_scores_ref, dict) and "artifact_key" in y_scores_ref:
                y_scores = _load_artifact(res, y_scores_ref["artifact_key"])
            else:
                y_scores = np.asarray(y_scores_ref) if y_scores_ref is not None else None

            if x_scores is None or y_scores is None:
                continue
            if x_scores_sum is None:
                x_scores_sum = np.zeros((n_subjects, x_scores.shape[1]), dtype=float)
                y_scores_sum = np.zeros((n_subjects, y_scores.shape[1]), dtype=float)

            x_scores_sum[train_idx] += x_scores
            y_scores_sum[train_idx] += y_scores
            score_counts[train_idx] += 1

            test_idx = fold.get("test_out_idx")
            if test_idx is None:
                continue
            test_idx = np.asarray(test_idx, dtype=int).reshape(-1)

            x_test_ref = fold.get("test_scores_X")
            y_test_ref = fold.get("test_scores_Y")
            if isinstance(x_test_ref, dict) and "artifact_key" in x_test_ref:
                x_test_scores = _load_artifact(res, x_test_ref["artifact_key"])
            else:
                x_test_scores = np.asarray(x_test_ref) if x_test_ref is not None else None
            if isinstance(y_test_ref, dict) and "artifact_key" in y_test_ref:
                y_test_scores = _load_artifact(res, y_test_ref["artifact_key"])
            else:
                y_test_scores = np.asarray(y_test_ref) if y_test_ref is not None else None

            if x_test_scores is None or y_test_scores is None:
                continue
            if x_test_sum is None:
                x_test_sum = np.zeros((n_subjects, x_test_scores.shape[1]), dtype=float)
                y_test_sum = np.zeros((n_subjects, y_test_scores.shape[1]), dtype=float)
            x_test_sum[test_idx] += x_test_scores
            y_test_sum[test_idx] += y_test_scores
            test_counts[test_idx] += 1

        if x_scores_sum is None or y_scores_sum is None:
            continue

        counts = np.maximum(score_counts, 1)
        x_mean = x_scores_sum / counts
        y_mean = y_scores_sum / counts
        run_subject_scores_x_train.append(x_mean)
        run_subject_scores_y_train.append(y_mean)

        if x_test_sum is not None and y_test_sum is not None:
            t_counts = np.maximum(test_counts, 1)
            x_test_mean = x_test_sum / t_counts
            y_test_mean = y_test_sum / t_counts
            run_subject_scores_x_test.append(x_test_mean)
            run_subject_scores_y_test.append(y_test_mean)

    if not all_fold_scores:
        raise RuntimeError("No real fold scores found.")

    max_components = max(arr.size for arr in all_fold_scores)
    score_mat = np.full((len(all_fold_scores), max_components), np.nan, dtype=float)
    for i, arr in enumerate(all_fold_scores):
        n = min(arr.size, max_components)
        score_mat[i, :n] = arr[:n]

    mean_scores = np.nanmean(score_mat, axis=0)
    median_scores = np.nanmedian(score_mat, axis=0)

    out_csv = output_dir / "real_scores_summary.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("component,score_mean,score_median,n_folds\n")
        for i in range(max_components):
            f.write(f"{i+1},{mean_scores[i]:.6f},{median_scores[i]:.6f},{score_mat.shape[0]}\n")

    agg_npz = output_dir / "real_loadings_scores_summary.npz"
    np.savez_compressed(
        agg_npz,
        scores_mean=mean_scores,
        scores_median=median_scores,
    )

    if all_fold_x_loadings and all_fold_y_loadings:
        x_stack = np.stack(all_fold_x_loadings, axis=0)
        y_stack = np.stack(all_fold_y_loadings, axis=0)
        x_agg = np.nanmean(x_stack, axis=0)
        y_agg = np.nanmean(y_stack, axis=0)
        np.save(output_dir / "real_x_loadings_summary.npy", x_agg)
        np.save(output_dir / "real_y_loadings_summary.npy", y_agg)

    if run_subject_scores_x_train and run_subject_scores_y_train:
        # mean across real runs
        x_runs = np.stack(run_subject_scores_x_train, axis=0)
        y_runs = np.stack(run_subject_scores_y_train, axis=0)
        x_mean_all = np.nanmean(x_runs, axis=0)
        y_mean_all = np.nanmean(y_runs, axis=0)
        np.savez_compressed(
            output_dir / "real_subject_train_scores.npz",
            X_scores_mean=x_mean_all,
            Y_scores_mean=y_mean_all,
        )

    if run_subject_scores_x_test and run_subject_scores_y_test:
        x_runs = np.stack(run_subject_scores_x_test, axis=0)
        y_runs = np.stack(run_subject_scores_y_test, axis=0)
        x_mean_all = np.nanmean(x_runs, axis=0)
        y_mean_all = np.nanmean(y_runs, axis=0)
        np.savez_compressed(
            output_dir / "real_subject_test_scores.npz",
            X_scores_mean=x_mean_all,
            Y_scores_mean=y_mean_all,
        )

    print(f"Saved: {out_csv}")
    print(f"Saved: {agg_npz}")
if __name__ == "__main__":
    main()
