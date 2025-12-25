import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import load_results


def _iter_perm_results(results_root: Path, atlas: str | None, model_type: str | None):
    base = results_root / "perm"
    if atlas is not None:
        base = base / atlas
    if model_type is not None:
        base = base / model_type
    if not base.exists():
        return
    for p in base.glob("**/seed_*/result.json"):
        yield p
    for p in base.glob("**/seed_*/result.npz"):
        yield p


def _load_real_scores(results_root: Path, atlas: str | None) -> np.ndarray | None:
    if atlas is None:
        return None
    summary_path = results_root / "summary" / atlas / "real_loadings_scores_summary.npz"
    if not summary_path.exists():
        return None
    try:
        data = np.load(summary_path)
        scores = data.get("scores_median")
        if scores is None:
            scores = data.get("scores_mean")
        return np.asarray(scores, dtype=float).reshape(-1)
    except Exception:
        return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--atlas", type=str, default=None)
    ap.add_argument("--model_type", type=str, default=None)
    ap.add_argument("--output_csv", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    perm_scores = []
    real_scores = _load_real_scores(results_root, args.atlas)

    for path in _iter_perm_results(results_root, args.atlas, args.model_type):
        try:
            res = load_results(path)
        except Exception:
            continue

        step_scores = res.get("stepwise_scores")
        if step_scores is None:
            continue
        step_scores = np.asarray(step_scores, dtype=float).reshape(-1)
        if step_scores.size == 0:
            continue
        perm_scores.append(step_scores)

        if real_scores is None:
            real_from_perm = res.get("real_scores_mean")
            if real_from_perm is not None:
                real_scores = np.asarray(real_from_perm, dtype=float).reshape(-1)

    if not perm_scores:
        raise RuntimeError("No permutation stepwise scores found.")
    if real_scores is None:
        raise RuntimeError("Real scores not found; run real summary first.")

    n_components = max(arr.size for arr in perm_scores)
    perm_mat = np.full((len(perm_scores), n_components), np.nan, dtype=float)
    for i, arr in enumerate(perm_scores):
        n = min(arr.size, n_components)
        perm_mat[i, :n] = arr[:n]

    real_scores = np.asarray(real_scores, dtype=float).reshape(-1)
    if real_scores.size < n_components:
        pad = np.full((n_components - real_scores.size,), np.nan, dtype=float)
        real_scores = np.concatenate([real_scores, pad])

    out_csv = Path(args.output_csv) if args.output_csv else results_root / "summary" / "perm_stepwise_pvalues.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("component,real_score,perm_mean,perm_std,n_perm,p_right\n")
        for i in range(n_components):
            perm_vals = perm_mat[:, i]
            perm_vals = perm_vals[np.isfinite(perm_vals)]
            n_perm = int(perm_vals.size)
            if n_perm == 0:
                continue
            real_score = real_scores[i]
            count_ge = int(np.sum(perm_vals >= real_score))
            p_right = float((count_ge + 1) / (n_perm + 1))
            f.write(
                f"{i+1},{real_score:.6f},{np.nanmean(perm_vals):.6f},{np.nanstd(perm_vals):.6f},{n_perm},{p_right:.6f}\n"
            )

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
