#!/usr/bin/env python3
"""
Exploratory visualizations for behavioral task metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACC_PRIORITY_SUFFIXES = [
    "_ACC",
    "_Overall_ACC",
    "_NoGo_ACC",
    "_Go_ACC",
    "_Stop_ACC",
    "_Switch_ACC",
    "_Repeat_ACC",
    "_Switch_Cost_ACC",
    "_Contrast_ACC",
    "_Congruent_ACC",
    "_Incongruent_ACC",
]

RT_PRIORITY_SUFFIXES = [
    "_RT_Mean",
    "_Mean_RT",
    "_Go_RT_Mean",
    "_RT_SD",
    "_RT",
    "_Contrast_RT",
    "_Switch_Cost_RT",
]

DPRIME_SUFFIXES = [
    "_dprime",
]

EXTRA_DPRIME_METRICS = [
    "SST_SSRT",
    "SST_Mean_SSD",
]

CONTRAST_KEYS = ["_Contrast_RT", "_Contrast_ACC"]
SWITCH_COST_KEYS = ["_Switch_Cost_RT", "_Switch_Cost_ACC"]


def setup_logging(log_path):
    """Configure logging to file and stdout."""
    if not log_path:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_task_prefixes(config_path):
    """Load task prefixes from config.json."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    prefixes = config.get("behavioral", {}).get("task_prefixes", [])
    if not prefixes:
        raise ValueError("Missing behavioral.task_prefixes in config.json")
    return prefixes


def pick_acc_metric(task_prefix, acc_cols):
    """Pick the primary ACC metric for a task."""
    candidates = [c for c in acc_cols if c.startswith(task_prefix + "_")]
    if not candidates:
        return "", []

    for suffix in ACC_PRIORITY_SUFFIXES:
        target = task_prefix + suffix
        if target in candidates:
            return target, candidates
    return candidates[0], candidates


def pick_metric_by_suffix(task_prefix, columns, suffixes):
    """Pick the primary metric by suffix priority."""
    candidates = [c for c in columns if c.startswith(task_prefix + "_")]
    if not candidates:
        return "", []

    for suffix in suffixes:
        target = task_prefix + suffix
        if target in candidates:
            return target, candidates
    return candidates[0], candidates


def summarize_acc(df, task_prefixes):
    """Select ACC metrics per task and summarize."""
    acc_cols = [c for c in df.columns if c.endswith("_ACC")]

    selections = []
    summary_rows = []

    for task in task_prefixes:
        metric, candidates = pick_acc_metric(task, acc_cols)
        selections.append(
            {
                "Task": task,
                "Selected_ACC": metric,
                "Available_ACC": ";".join(candidates),
            }
        )

        if not metric:
            summary_rows.append(
                {
                    "Task": task,
                    "Metric": "",
                    "N": 0,
                    "Missing": len(df),
                    "Mean": np.nan,
                    "Std": np.nan,
                    "Median": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                }
            )
            continue

        values = pd.to_numeric(df[metric], errors="coerce")
        summary_rows.append(
            {
                "Task": task,
                "Metric": metric,
                "N": int(values.notna().sum()),
                "Missing": int(values.isna().sum()),
                "Mean": round(float(values.mean()), 4) if values.notna().any() else np.nan,
                "Std": round(float(values.std(ddof=0)), 4) if values.notna().any() else np.nan,
                "Median": round(float(values.median()), 4) if values.notna().any() else np.nan,
                "Min": round(float(values.min()), 4) if values.notna().any() else np.nan,
                "Max": round(float(values.max()), 4) if values.notna().any() else np.nan,
            }
        )

    selection_df = pd.DataFrame(selections)
    summary_df = pd.DataFrame(summary_rows)
    return selection_df, summary_df


def summarize_metric(df, task_prefixes, columns, suffixes, metric_label):
    """Select a metric per task and summarize."""
    selections = []
    summary_rows = []

    for task in task_prefixes:
        metric, candidates = pick_metric_by_suffix(task, columns, suffixes)
        selections.append(
            {
                "Task": task,
                "Selected": metric,
                "Available": ";".join(candidates),
                "Metric_Type": metric_label,
            }
        )

        if not metric:
            summary_rows.append(
                {
                    "Task": task,
                    "Metric": "",
                    "Metric_Type": metric_label,
                    "N": 0,
                    "Missing": len(df),
                    "Mean": np.nan,
                    "Std": np.nan,
                    "Median": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                }
            )
            continue

        values = pd.to_numeric(df[metric], errors="coerce")
        summary_rows.append(
            {
                "Task": task,
                "Metric": metric,
                "Metric_Type": metric_label,
                "N": int(values.notna().sum()),
                "Missing": int(values.isna().sum()),
                "Mean": round(float(values.mean()), 4) if values.notna().any() else np.nan,
                "Std": round(float(values.std(ddof=0)), 4) if values.notna().any() else np.nan,
                "Median": round(float(values.median()), 4) if values.notna().any() else np.nan,
                "Min": round(float(values.min()), 4) if values.notna().any() else np.nan,
                "Max": round(float(values.max()), 4) if values.notna().any() else np.nan,
            }
        )

    selection_df = pd.DataFrame(selections)
    summary_df = pd.DataFrame(summary_rows)
    return selection_df, summary_df


def plot_acc_grid(df, selection_df, output_path, title):
    """Plot 4x4 grid of ACC distributions."""
    tasks = selection_df["Task"].tolist()
    metrics = selection_df["Selected_ACC"].tolist()
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    for idx, (task, metric) in enumerate(zip(tasks, metrics)):
        ax = axes[idx // 4][idx % 4]
        if not metric:
            ax.set_title(f"{task} (no ACC)")
            ax.axis("off")
            continue

        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        ax.hist(values, bins=20, color="#4C72B0", alpha=0.8)
        ax.set_title(task)
        ax.set_xlabel("ACC")
        ax.set_ylabel("Count")
        if not values.empty and values.max() <= 1.2:
            ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(tasks), 16):
        axes[idx // 4][idx % 4].axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_grid(df, selection_df, output_path, title, xlabel, grid_shape=None):
    """Plot metric distributions with optional grid size."""
    selection_df = selection_df.copy()
    selection_df = selection_df[selection_df["Selected"] != ""]
    tasks = selection_df["Task"].tolist()
    metrics = selection_df["Selected"].tolist()

    if not tasks:
        return

    if grid_shape is None:
        n = len(tasks)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.atleast_2d(axes)

    for idx, (task, metric) in enumerate(zip(tasks, metrics)):
        ax = axes[idx // cols][idx % cols]
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        ax.hist(values, bins=20, color="#4C72B0", alpha=0.8)
        ax.set_title(task)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(tasks), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_list(df, metrics, output_path, title, xlabel, grid_shape=None):
    """Plot distributions for an explicit metric list."""
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    if grid_shape is None:
        n = len(metrics)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.atleast_2d(axes)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        ax.hist(values, bins=20, color="#4C72B0", alpha=0.8)
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(metrics), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory plots for behavioral ACC metrics."
    )
    parser.add_argument(
        "--behavioral_csv",
        default=r"d:\code\data_driven_EF\data\EFNY\table\demo\EFNY_behavioral_data.csv",
        help="Behavioral data CSV path.",
    )
    parser.add_argument(
        "--config",
        default=r"d:\code\data_driven_EF\src\models\config.json",
        help="Config JSON containing task prefixes.",
    )
    parser.add_argument(
        "--output_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\figures\metric_exploration",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--log",
        default=r"d:\code\data_driven_EF\data\EFNY\log\behavior_data\metric_exploration.log",
        help="Log file path.",
    )
    parser.add_argument(
        "--summary_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\metric_exploration",
        help="Output directory for CSV summaries.",
    )

    args = parser.parse_args()
    setup_logging(args.log)

    if not os.path.exists(args.behavioral_csv):
        logging.error("Behavioral CSV not found: %s", args.behavioral_csv)
        return 1

    task_prefixes = load_task_prefixes(args.config)
    df = pd.read_csv(args.behavioral_csv)

    selection_df, summary_df = summarize_acc(df, task_prefixes)

    rt_cols = [c for c in df.columns if "_RT" in c]
    rt_selection_df, rt_summary_df = summarize_metric(
        df, task_prefixes, rt_cols, RT_PRIORITY_SUFFIXES, "RT"
    )

    dprime_cols = [c for c in df.columns if c.endswith("_dprime")]
    dprime_selection_df, dprime_summary_df = summarize_metric(
        df, task_prefixes, dprime_cols, DPRIME_SUFFIXES, "dprime"
    )

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    selection_path = summary_dir / "acc_metric_selection.csv"
    summary_path = summary_dir / "acc_summary.csv"
    rt_selection_path = summary_dir / "rt_metric_selection.csv"
    rt_summary_path = summary_dir / "rt_summary.csv"
    dprime_selection_path = summary_dir / "dprime_metric_selection.csv"
    dprime_summary_path = summary_dir / "dprime_summary.csv"
    contrast_switch_path = summary_dir / "contrast_switch_metric_selection.csv"

    selection_df.to_csv(selection_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    rt_selection_df.to_csv(rt_selection_path, index=False, encoding="utf-8")
    rt_summary_df.to_csv(rt_summary_path, index=False, encoding="utf-8")
    dprime_selection_df.to_csv(dprime_selection_path, index=False, encoding="utf-8")
    dprime_summary_df.to_csv(dprime_summary_path, index=False, encoding="utf-8")
    pd.DataFrame({"Metric": []}).to_csv(contrast_switch_path, index=False, encoding="utf-8")

    logging.info("Selection summary saved to: %s", selection_path)
    logging.info("ACC summary saved to: %s", summary_path)
    logging.info("RT selection saved to: %s", rt_selection_path)
    logging.info("RT summary saved to: %s", rt_summary_path)
    logging.info("dprime selection saved to: %s", dprime_selection_path)
    logging.info("dprime summary saved to: %s", dprime_summary_path)
    logging.info("Contrast/Switch selection saved to: %s", contrast_switch_path)

    output_dir = Path(args.output_dir)
    plot_acc_grid(
        df,
        selection_df,
        output_dir / "acc_distribution_grid.png",
        "ACC Distributions",
    )
    plot_metric_grid(
        df,
        rt_selection_df.rename(columns={"Selected": "Selected"}),
        output_dir / "rt_distribution_grid.png",
        "RT Distributions",
        "RT",
        grid_shape=(4, 4),
    )
    plot_metric_list(
        df,
        dprime_selection_df["Selected"].tolist() + EXTRA_DPRIME_METRICS,
        output_dir / "dprime_distribution_grid.png",
        "dprime + SSRT/SSD Distributions",
        "Metric Value",
    )

    contrast_switch_metrics = [
        c for c in df.columns if any(key in c for key in CONTRAST_KEYS + SWITCH_COST_KEYS)
    ]
    if contrast_switch_metrics:
        contrast_switch_metrics = sorted(contrast_switch_metrics)
        pd.DataFrame({"Metric": contrast_switch_metrics}).to_csv(
            contrast_switch_path, index=False, encoding="utf-8"
        )
        plot_metric_list(
            df,
            contrast_switch_metrics,
            output_dir / "contrast_switch_distribution_grid.png",
            "Contrast + Switch Cost Distributions",
            "Metric Value",
        )

    logging.info("Figures saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
