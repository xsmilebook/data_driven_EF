import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine"])
    return ap.parse_args()


def _resolve_artifact_path(result_dir: Path, artifact_path: str | None) -> Path | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if path.exists():
        return path
    rel = result_dir / artifact_path
    if rel.exists():
        return rel
    fallback = result_dir / "artifacts" / path.name
    if fallback.exists():
        return fallback
    return None


def _load_artifact(result: dict, result_dir: Path, ref) -> np.ndarray | None:
    if isinstance(ref, dict) and "artifact_key" in ref:
        key = ref["artifact_key"]
        entry = result.get("artifacts", {}).get(key)
        if not entry:
            return None
        artifact_path = entry.get("path")
    else:
        artifact_path = ref

    resolved = _resolve_artifact_path(result_dir, artifact_path)
    if resolved is None:
        return None
    return np.load(resolved)


def _collect_loadings(result: dict, result_dir: Path, kind: str) -> list[np.ndarray]:
    outer_folds = result.get("cv_results", {}).get("outer_fold_results", [])
    loadings = []
    for fold in outer_folds:
        ref = fold.get(f"{kind}_loadings")
        arr = _load_artifact(result, result_dir, ref)
        if arr is None:
            return []
        loadings.append(arr)
    return loadings


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * s[:2]


def _plot_embedding(coords: np.ndarray, labels: list[tuple[int, int]], n_components: int, output_path: Path, title: str):
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_components, 1)))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for idx, (fold, comp) in enumerate(labels):
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            color=colors[comp],
            marker=markers[fold % len(markers)],
            s=70,
            edgecolor="black",
            linewidths=0.5,
            alpha=0.9,
        )
        ax.text(coords[idx, 0], coords[idx, 1], f"F{fold}", fontsize=8, ha="left", va="bottom")

    handles = []
    for comp in range(n_components):
        handles.append(
            plt.Line2D(
                [],
                [],
                color=colors[comp],
                marker="o",
                linestyle="None",
                markersize=8,
                label=f"C{comp + 1}",
            )
        )
    ax.legend(handles=handles, title="Component", loc="best")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_similarity_heatmap(sim: np.ndarray, labels: list[tuple[int, int]], output_path: Path, title: str):
    tick_labels = [f"F{fold}_C{comp + 1}" for fold, comp in labels]
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_similarity_csv(sim: np.ndarray, labels: list[tuple[int, int]], output_path: Path):
    header = ["label"] + [f"F{fold}_C{comp + 1}" for fold, comp in labels]
    lines = [",".join(header)]
    for i, (fold, comp) in enumerate(labels):
        row = [f"F{fold}_C{comp + 1}"] + [f"{sim[i, j]:.6f}" for j in range(sim.shape[1])]
        lines.append(",".join(row))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_component_summary(sim: np.ndarray, labels: list[tuple[int, int]], n_components: int, output_path: Path):
    lines = ["component,mean_cosine_similarity,std_cosine_similarity,n_pairs"]
    for comp in range(n_components):
        idx = [i for i, (_, c) in enumerate(labels) if c == comp]
        if len(idx) < 2:
            lines.append(f"{comp + 1},nan,nan,0")
            continue
        block = sim[np.ix_(idx, idx)]
        tri = block[np.triu_indices_from(block, k=1)]
        lines.append(f"{comp + 1},{np.mean(tri):.6f},{np.std(tri):.6f},{tri.size}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_for_kind(kind: str, result: dict, result_dir: Path, output_dir: Path):
    loadings = _collect_loadings(result, result_dir, kind)
    if not loadings:
        raise RuntimeError(f"Missing {kind}_loadings for one or more folds.")

    n_folds = len(loadings)
    n_components = loadings[0].shape[1]
    vectors = []
    labels = []
    for fold_idx, arr in enumerate(loadings):
        if arr.shape[1] != n_components:
            raise ValueError("Inconsistent component counts across folds.")
        for comp in range(n_components):
            vectors.append(arr[:, comp])
            labels.append((fold_idx, comp))

    vectors = np.stack(vectors, axis=0)
    sim = _cosine_similarity_matrix(vectors)
    coords = _pca_2d(vectors)

    kind_upper = kind.upper()
    _plot_embedding(
        coords,
        labels,
        n_components,
        output_dir / f"{kind}_loadings_pca.png",
        f"{kind_upper} loadings PCA (folds={n_folds}, components={n_components})",
    )
    _plot_similarity_heatmap(
        sim,
        labels,
        output_dir / f"{kind}_loadings_similarity_heatmap.png",
        f"{kind_upper} loadings cosine similarity",
    )
    _write_similarity_csv(sim, labels, output_dir / f"{kind}_loadings_cosine_similarity.csv")
    _write_component_summary(sim, labels, n_components, output_dir / f"{kind}_loadings_component_similarity.csv")


def run_for_result_dir(result_dir: Path, output_dir: Path | None = None):
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    result_json = result_dir / "result.json"
    if not result_json.exists():
        raise FileNotFoundError(f"result.json not found: {result_json}")

    out_dir = Path(output_dir) if output_dir else result_dir / "loadings_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    with result_json.open("r", encoding="utf-8") as f:
        result = json.load(f)

    _run_for_kind("x", result, result_dir, out_dir)
    _run_for_kind("y", result, result_dir, out_dir)
    return out_dir


def main():
    args = parse_args()
    out_dir = run_for_result_dir(Path(args.result_dir), Path(args.output_dir) if args.output_dir else None)
    print(f"Saved figures and tables to: {out_dir}")


if __name__ == "__main__":
    main()
