import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _parse_corrs(s: str) -> list[float]:
    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in list(s)]
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [float(x) for x in v]
    except Exception:
        pass
    return []


def _dedup_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows coming from both result.json and result.npz.

    Prefer .json over .npz if both exist for the same run.
    """

    if df.empty:
        return df

    keys = ["analysis_type", "atlas", "model_type", "seed", "task_id", "permutation_seed"]
    for k in keys:
        if k not in df.columns:
            df[k] = np.nan

    rp = df.get("result_path")
    if rp is None:
        df["_prefer"] = 0
    else:
        rp = rp.astype(str)
        df["_prefer"] = np.where(rp.str.endswith(".json"), 2, np.where(rp.str.endswith(".npz"), 1, 0))

    df = df.sort_values(keys + ["_prefer"], ascending=True)
    df = df.drop_duplicates(subset=keys, keep="last")
    df = df.drop(columns=["_prefer"], errors="ignore")
    return df


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        corrs = _parse_corrs(r.get("canonical_correlations"))
        for i, c in enumerate(corrs, start=1):
            rows.append(
                {
                    "analysis_type": r.get("analysis_type"),
                    "atlas": r.get("atlas"),
                    "model_type": r.get("model_type"),
                    "seed": r.get("seed"),
                    "task_id": r.get("task_id"),
                    "permutation_seed": r.get("permutation_seed"),
                    "component": i,
                    "corr": float(c),
                }
            )
    return pd.DataFrame(rows)


def _empirical_p_ge(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    # +1 smoothing
    return float((np.sum(null_values >= observed) + 1) / (null_values.size + 1))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--out_name", type=str, default="real_component_median_vs_perm_pvalues.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary_csv not found: {summary_csv}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df = _dedup_runs(df)

    long_df = _to_long(df)
    if long_df.empty:
        out_path = out_dir / args.out_name
        pd.DataFrame([]).to_csv(out_path, index=False)
        print(f"Wrote empty: {out_path}")
        return

    real = long_df[long_df["analysis_type"] == "real"].copy()
    perm = long_df[long_df["analysis_type"] == "perm"].copy()

    real_summary = (
        real.groupby(["atlas", "model_type", "component"], dropna=False)["corr"]
        .agg(real_median="median", n_real="count")
        .reset_index()
    )

    if perm.empty:
        out = real_summary
        out["perm_median"] = np.nan
        out["n_perm"] = 0
        out["p_value_perm_ge_real_median"] = np.nan
    else:
        perm_summary = (
            perm.groupby(["atlas", "model_type", "component"], dropna=False)["corr"]
            .agg(perm_median="median", n_perm="count")
            .reset_index()
        )

        out = real_summary.merge(perm_summary, on=["atlas", "model_type", "component"], how="left")

        # Compute p-values per group
        pvals = []
        for _, row in out.iterrows():
            atlas = row["atlas"]
            model = row["model_type"]
            comp = row["component"]
            obs = row["real_median"]
            null_vals = perm[(perm["atlas"] == atlas) & (perm["model_type"] == model) & (perm["component"] == comp)]["corr"].to_numpy()
            pvals.append(_empirical_p_ge(float(obs), null_vals))
        out["p_value_perm_ge_real_median"] = pvals

    out = out.sort_values(["atlas", "model_type", "component"], na_position="last")

    out_path = out_dir / args.out_name
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
