import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import load_results


def _corr_vector(result: dict) -> np.ndarray:
    for key in ("mean_canonical_correlations", "canonical_correlations"):
        v = result.get(key)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size:
            return arr
    return np.asarray([], dtype=float)


def _score_from_corrs(corrs: np.ndarray, score_mode: str) -> float:
    if corrs.size == 0:
        return float("nan")

    if score_mode == "first_component":
        return float(corrs[0])
    if score_mode == "mean_all":
        return float(np.nanmean(corrs))
    if score_mode == "vector3":
        return float(np.linalg.norm(corrs))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _glob_files(base: Path, glob_pattern: str):
    try:
        yield from base.glob(glob_pattern)
    except Exception:
        return


def iter_result_files(results_root: Path, analysis_type: str, atlas: str | None, model_type: str | None):
    base = results_root / analysis_type
    if not base.exists():
        return

    if atlas is not None:
        base = base / atlas
        if not base.exists():
            return

    if model_type is None:
        json_glob = "**/seed_*/result.json"
        npz_glob = "**/seed_*/result.npz"
    else:
        json_glob = f"**/{model_type}/seed_*/result.json"
        npz_glob = f"**/{model_type}/seed_*/result.npz"

    seen: set[Path] = set()
    for p in _glob_files(base, json_glob):
        if p not in seen:
            seen.add(p)
            yield p
    for p in _glob_files(base, npz_glob):
        if p not in seen:
            seen.add(p)
            yield p


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--analysis_type", type=str, default="both", choices=["real", "perm", "both"])
    ap.add_argument("--atlas", type=str, default=None)
    ap.add_argument("--model_type", type=str, default=None)
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument(
        "--score_mode",
        type=str,
        default="first_component",
        choices=["first_component", "mean_all", "vector3"],
    )
    return ap.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    analysis_types = [args.analysis_type] if args.analysis_type != "both" else ["real", "perm"]

    rows = []
    for atype in analysis_types:
        for path in iter_result_files(results_root, atype, args.atlas, args.model_type):
            try:
                res = load_results(path)
            except Exception:
                continue

            corrs = _corr_vector(res)
            score = _score_from_corrs(corrs, args.score_mode)

            seed = None
            try:
                seed = path.parent.name.replace("seed_", "")
            except Exception:
                seed = None

            model_type = None
            try:
                model_type = path.parent.parent.name
            except Exception:
                model_type = None

            atlas = None
            try:
                atlas = path.parent.parent.parent.name
            except Exception:
                atlas = None

            meta = res.get("metadata", {}) if isinstance(res, dict) else {}

            rows.append(
                {
                    "analysis_type": atype,
                    "atlas": atlas,
                    "model_type": model_type,
                    "seed": seed,
                    "task_id": meta.get("task_id", res.get("task_id")),
                    "permutation_seed": meta.get("permutation_seed", res.get("permutation_seed")),
                    "score_mean_corr": score,
                    "n_components": int(corrs.size),
                    "canonical_correlations": json.dumps(corrs.tolist()) if corrs.size else "[]",
                    "result_path": str(path),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["analysis_type", "atlas", "model_type", "seed"], na_position="last")

    if args.output_csv is None:
        out = results_root / "summary_scores.csv"
    else:
        out = Path(args.output_csv)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote: {out}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
