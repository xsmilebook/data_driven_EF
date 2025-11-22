import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    return df

def load_task_cfg(task_csv: Path) -> tuple[dict, dict]:
    try:
        df = pd.read_csv(task_csv, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(task_csv, dtype=str, keep_default_na=False, engine="python", on_bad_lines="warn")
    df.columns = [c.strip() for c in df.columns]
    dims = {}
    valids = {}
    for _, r in df.iterrows():
        task = str(r.get("Task", "")).strip()
        dim = str(r.get("dim", "")).strip()
        if not task:
            continue
        dims[task] = dim
        ms = set()
        for m in ["d_prime", "SSRT", "ACC", "Reaction_Time", "Constrast_ACC", "Constrast_RT", "Switch_Cost"]:
            v = str(r.get(m, "")).strip().lower()
            if v in ("true", "1", "yes"):
                ms.add(m)
        valids[task] = ms
    return dims, valids

def select_numeric_columns(df: pd.DataFrame, min_ratio: float) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ["subject_code", "file_name"]]
    num = {}
    n = len(df)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = np.count_nonzero(~np.isnan(s)) / n if n > 0 else 0.0
        if valid > min_ratio:
            num[c] = s
    return pd.DataFrame(num)

def filter_by_task_validity(df: pd.DataFrame, dims: dict, valids: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    kept = {}
    labels = []
    groups = []
    for c in df.columns:
        if "_" not in c:
            continue
        task, metric = c.split("_", 1)
        mname = metric
        if task in valids and mname in valids[task]:
            kept[c] = df[c]
            labels.append(c)
            groups.append(dims.get(task, ""))
    out = pd.DataFrame(kept)
    return out, labels, groups

def order_by_groups(df: pd.DataFrame, labels: list[str], groups: list[str], order: list[str]) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    items = list(zip(labels, groups))
    ordered = []
    for g in order:
        for lab, grp in items:
            if grp == g:
                ordered.append(lab)
    for lab, grp in items:
        if grp not in order:
            ordered.append(lab)
    df2 = df[ordered]
    grp2 = []
    for lab in ordered:
        task = lab.split("_", 1)[0]
        grp2.append(groups[labels.index(lab)])
    boundaries = []
    count = 0
    for g in order:
        k = sum(1 for x in grp2 if x == g)
        count += k
        if k > 0:
            boundaries.append(count)
    return df2, ordered, grp2, boundaries

def compute_pairwise_corr(df: pd.DataFrame, method: str, min_pair_ratio: float) -> pd.DataFrame:
    n = len(df)
    cols = list(df.columns)
    mat = np.full((len(cols), len(cols)), np.nan, dtype=float)
    for i, a in enumerate(cols):
        sa = df[a]
        for j, b in enumerate(cols):
            sb = df[b]
            mask = (~sa.isna()) & (~sb.isna())
            if n > 0 and (mask.sum() / n) >= min_pair_ratio:
                mat[i, j] = sa[mask].corr(sb[mask], method=method)
    return pd.DataFrame(mat, index=cols, columns=cols)

def plot_heatmap(mat: pd.DataFrame, out_path: Path, title: str, boundaries: list[int]) -> None:
    labels = list(mat.columns)
    arr = mat.to_numpy()
    masked = np.ma.array(arr, mask=np.isnan(arr))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")
    size = max(6, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="black", linewidth=2)
        ax.axhline(b - 0.5, color="black", linewidth=2)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    default_csv = base / "data" / "EFNY" / "table" / "EFNY_metrics.csv"
    default_task = base / "data" / "EFNY" / "table" / "EFNY_task.csv"
    default_png = base / "data" / "EFNY" / "table" / "EFNY_metrics_similarity_heatmap.png"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--task-csv", default=str(default_task))
    parser.add_argument("--out-png", default=str(default_png))
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"])
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--min-pair-ratio", type=float, default=0.5)
    args = parser.parse_args()
    csv_path = Path(args.csv)
    task_path = Path(args.task_csv)
    if not csv_path.exists():
        print(f"Not found: {csv_path}", file=sys.stderr)
        return
    if not task_path.exists():
        print(f"Not found: {task_path}", file=sys.stderr)
        return
    dims, valids = load_task_cfg(task_path)
    df = load_metrics(csv_path)
    num_df = select_numeric_columns(df, args.min_valid_ratio)
    filt_df, labels, groups = filter_by_task_validity(num_df, dims, valids)
    if filt_df.shape[1] == 0:
        print("No valid metric columns", file=sys.stderr)
        return
    ordered_df, ordered_labels, ordered_groups, boundaries = order_by_groups(filt_df, labels, groups, ["Inhibition", "Working_Memory", "Shifting"])
    corr = compute_pairwise_corr(ordered_df, args.method, args.min_pair_ratio)
    corr = corr.loc[ordered_labels, ordered_labels]
    out_path = Path(args.out_png)
    if not out_path.is_absolute():
        out_path = csv_path.parent / out_path.name
    plot_heatmap(corr, out_path, f"EFNY Metrics Similarity ({args.method})", boundaries)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()