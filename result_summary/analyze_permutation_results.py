#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import load_results
from src.models.evaluation import PermutationTester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="adaptive_pls",
    )
    parser.add_argument(
        "--max_task_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_npz",
        action="store_true",
    )
    return parser.parse_args()


def find_latest_result_file(task_dir: Path, use_npz: bool) -> Path | None:
    if use_npz:
        files = sorted(task_dir.glob("*.npz"))
    else:
        files = sorted(task_dir.glob("*.json"))
        if not files:
            files = sorted(task_dir.glob("*.npz"))
    if not files:
        return None
    return files[-1]


def load_canonical_correlations(path: Path) -> np.ndarray:
    res = load_results(path)
    if "canonical_correlations" in res:
        return np.asarray(res["canonical_correlations"], dtype=float)
    raise KeyError(f"canonical_correlations not found in {path}")


def load_vector3_statistic(path: Path) -> float:
    res = load_results(path)
    if "cv_statistic_vector3" in res:
        return float(res["cv_statistic_vector3"])
    if "canonical_correlations" in res:
        corrs = np.asarray(res["canonical_correlations"], dtype=float)
        return float(np.linalg.norm(corrs))
    raise KeyError(f"Neither cv_statistic_vector3 nor canonical_correlations found in {path}")


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    real_dir = results_root / f"task_0_{args.model_type}"
    if not real_dir.exists():
        raise FileNotFoundError(f"real-data directory not found: {real_dir}")

    real_file = find_latest_result_file(real_dir, args.use_npz)
    if real_file is None:
        raise FileNotFoundError(f"no result file found in {real_dir}")

    observed_corrs = load_canonical_correlations(real_file)
    observed_vector3 = load_vector3_statistic(real_file)

    perm_corrs = []
    perm_vector3 = []
    used_task_ids = []
    for task_id in range(1, args.max_task_id + 1):
        task_dir = results_root / f"task_{task_id}_{args.model_type}"
        if not task_dir.exists():
            continue
        perm_file = find_latest_result_file(task_dir, args.use_npz)
        if perm_file is None:
            continue
        try:
            cc = load_canonical_correlations(perm_file)
            v3 = load_vector3_statistic(perm_file)
        except Exception:
            continue
        if cc.shape != observed_corrs.shape:
            continue
        perm_corrs.append(cc)
        perm_vector3.append(v3)
        used_task_ids.append(task_id)

    if not perm_corrs:
        raise RuntimeError("no valid permutation results found")

    permuted_array = np.vstack(perm_corrs)
    permuted_vector3 = np.asarray(perm_vector3, dtype=float)

    tester = PermutationTester(n_permutations=len(perm_corrs), random_state=None)
    p_values = tester.calculate_p_values(observed_corrs, permuted_array)
    p_value_vector3 = tester.calculate_p_value_scalar(observed_vector3, permuted_vector3)

    perm_mean = permuted_array.mean(axis=0)
    perm_std = permuted_array.std(axis=0)

    n_components = observed_corrs.shape[0]
    lines = []
    header = [
        "Component",
        "Observed_r",
        "Perm_Mean_r",
        "Perm_Std_r",
        "p_value",
    ]
    lines.append(",".join(header))
    for i in range(n_components):
        row = [
            str(i + 1),
            f"{observed_corrs[i]:.6f}",
            f"{perm_mean[i]:.6f}",
            f"{perm_std[i]:.6f}",
            f"{p_values[i]:.6f}",
        ]
        lines.append(",".join(row))

    if args.output_prefix is None:
        output_prefix = f"perm_summary_{args.model_type}"
    else:
        output_prefix = args.output_prefix

    summary_path = results_root / f"{output_prefix}.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    vector3_path = results_root / f"{output_prefix}_vector3.csv"
    with vector3_path.open("w", encoding="utf-8") as f:
        f.write("Statistic,Observed,Perm_Mean,Perm_Std,p_value,n_permutations\n")
        f.write(
            "vector3,"
            f"{observed_vector3:.6f},"
            f"{float(np.mean(permuted_vector3)):.6f},"
            f"{float(np.std(permuted_vector3)):.6f},"
            f"{p_value_vector3:.6f},"
            f"{len(permuted_vector3)}\n"
        )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(1, n_components + 1)
        ax.bar(x - 0.2, observed_corrs, width=0.4, label="Observed")
        ax.bar(x + 0.2, perm_mean, width=0.4, label="Permutation mean")
        ax.errorbar(
            x + 0.2,
            perm_mean,
            yerr=perm_std,
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.set_xlabel("Component")
        ax.set_ylabel("Canonical correlation")
        ax.set_title(f"Observed vs permutation (model={args.model_type})")
        ax.legend()
        fig.tight_layout()
        fig_path = results_root / f"{output_prefix}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()

