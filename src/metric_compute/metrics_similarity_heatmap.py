import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TASK_DIMENSIONS = {
    "GNG": "inhibition",
    "FLANKER": "inhibition",
    "SST": "inhibition",
    "ColorStroop": "inhibition",
    "EmotionStroop": "inhibition",
    "CPT": "inhibition",
    "DCCS": "shifting",
    "DT": "shifting",
    "EmotionSwitch": "shifting",
    "KT": "updating",
    "oneback_number": "updating",
    "twoback_number": "updating",
    "oneback_spatial": "updating",
    "twoback_spatial": "updating",
    "oneback_emotion": "updating",
    "twoback_emotion": "updating",
}

SELECTED_METRICS = [
    "FLANKER_Contrast_RT",
    "FLANKER_Contrast_ACC",
    "FLANKER_RT_Mean",
    "SST_SSRT",
    "SST_Stop_ACC",
    "SST_Go_RT_Mean",
    "DT_Switch_Cost_RT",
    "DT_Switch_Cost_ACC",
    "DT_RT_Mean",
    "ColorStroop_Contrast_RT",
    "ColorStroop_Contrast_ACC",
    "ColorStroop_RT_Mean",
    "EmotionStroop_Contrast_RT",
    "EmotionStroop_Contrast_ACC",
    "EmotionStroop_RT_Mean",
    "CPT_dprime",
    "CPT_NoGo_ACC",
    "CPT_Go_RT_Mean",
    "EmotionSwitch_Switch_Cost_RT",
    "EmotionSwitch_Switch_Cost_ACC",
    "EmotionSwitch_RT_Mean",
    "oneback_number_dprime",
    "oneback_number_RT_Mean",
    "twoback_number_dprime",
    "twoback_number_RT_Mean",
    "oneback_spatial_dprime",
    "oneback_spatial_RT_Mean",
    "twoback_spatial_dprime",
    "twoback_spatial_RT_Mean",
    "KT_Overall_ACC",
    "KT_Mean_RT",
    "KT_RT_SD",
    "DCCS_Switch_Cost_RT",
    "DCCS_Switch_Cost_ACC",
    "DCCS_RT_Mean",
    "GNG_dprime",
    "GNG_NoGo_ACC",
    "GNG_Go_RT_Mean",
    "oneback_emotion_dprime",
    "oneback_emotion_RT_Mean",
    "twoback_emotion_dprime",
    "twoback_emotion_RT_Mean",
]

DIM_ORDER = ["inhibition", "shifting", "updating"]

HIGHER_BETTER_KEYS = [
    "ACC",
    "dprime",
]

LOWER_BETTER_KEYS = [
    "RT",
    "SSRT",
    "RT_Mean",
    "RT_SD",
    "Switch_Cost_RT",
    "Contrast_RT",
    "Go_RT_Mean",
    "Mean_RT",
]


def load_metrics(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="warn")
    return df


def get_task_prefix(metric_name: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        if metric_name.startswith(f"{prefix}_"):
            return prefix
    return ""


def select_metrics(df: pd.DataFrame, min_ratio: float) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    prefixes = sorted(TASK_DIMENSIONS.keys(), key=len, reverse=True)
    kept = {}
    labels = []
    groups = []
    missing = []
    n = len(df)

    for metric in SELECTED_METRICS:
        if metric not in df.columns:
            missing.append(metric)
            continue
        s = pd.to_numeric(df[metric], errors="coerce")
        valid_ratio = np.count_nonzero(~np.isnan(s)) / n if n > 0 else 0.0
        if valid_ratio <= min_ratio:
            continue
        task_prefix = get_task_prefix(metric, prefixes)
        group = TASK_DIMENSIONS.get(task_prefix, "")
        kept[metric] = s
        labels.append(metric)
        groups.append(group)

    out = pd.DataFrame(kept)
    return out, labels, groups, missing

def order_by_groups(
    df: pd.DataFrame,
    labels: list[str],
    groups: list[str],
    order: list[str],
) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    items = list(zip(labels, groups))
    ordered = []

    for g in order:
        group_items = [lab for lab, grp in items if grp == g]

        def desirability_key(label: str) -> tuple[int, str]:
            metric = label.split("_", 1)[1] if "_" in label else label
            metric_upper = metric.upper()

            if any(key.upper() in metric_upper for key in HIGHER_BETTER_KEYS):
                return (0, label)
            if any(key.upper() in metric_upper for key in LOWER_BETTER_KEYS):
                return (2, label)
            return (1, label)

        group_items.sort(key=desirability_key)
        ordered.extend(group_items)

    for lab, grp in items:
        if grp not in order:
            ordered.append(lab)

    df2 = df[ordered]
    grp2 = [groups[labels.index(lab)] for lab in ordered]
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
    
    # Increase size for better readability with more metrics
    size = max(8, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    
    # Create the heatmap
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    
    # Improve label readability
    ax.set_xticklabels(labels, rotation=90, fontsize=12, ha='center')
    ax.set_yticklabels(labels, fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Add boundary lines between cognitive domains
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="black", linewidth=2, alpha=0.7)
        ax.axhline(b - 0.5, color="black", linewidth=2, alpha=0.7)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    
    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with high DPI for better quality
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "data" / "EFNY" / "table" / "demo" / "EFNY_behavioral_data.csv"
    default_png = repo_root / "outputs" / "EFNY" / "figures" / "metrics" / "EFNY_metrics_similarity_heatmap.png"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--out-png", default=str(default_png))
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"])
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--min-pair-ratio", type=float, default=0.5)
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Not found: {csv_path}", file=sys.stderr)
        return
    df = load_metrics(csv_path)
    filt_df, labels, groups, missing = select_metrics(df, args.min_valid_ratio)
    if filt_df.shape[1] == 0:
        print("No valid metric columns", file=sys.stderr)
        return
    if missing:
        print("Missing metrics (not in CSV):", file=sys.stderr)
        for metric in missing:
            print(f"  - {metric}", file=sys.stderr)
    
    print(f"Found {len(labels)} valid metrics:")
    for i, label in enumerate(labels):
        print(f"  {i+1}: {label} (group: {groups[i]})")
    
    ordered_df, ordered_labels, ordered_groups, boundaries = order_by_groups(
        filt_df, labels, groups, DIM_ORDER
    )
    
    print(f"\nOrdered {len(ordered_labels)} metrics:")
    for i, label in enumerate(ordered_labels):
        print(f"  {i+1}: {label} (group: {ordered_groups[i]})")
    
    corr = compute_pairwise_corr(ordered_df, args.method, args.min_pair_ratio)
    corr = corr.loc[ordered_labels, ordered_labels]
    out_path = Path(args.out_png)
    if not out_path.is_absolute():
        out_path = csv_path.parent / out_path.name
    plot_heatmap(corr, out_path, f"EFNY Metrics Similarity ({args.method})", boundaries)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
