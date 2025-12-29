#!/usr/bin/env python3
"""
Plot task coverage summaries from EFNY_behavioral_data.csv outputs.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def apply_plot_style():
    """Increase font sizes for all plots."""
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def plot_task_missing(task_missing_df, ax):
    task_missing_df = task_missing_df.sort_values("Subjects_Missing_Data", ascending=True)
    ax.barh(task_missing_df["Task"], task_missing_df["Subjects_Missing_Data"], color="#4C72B0")
    ax.set_title("Missing Subjects by Task")
    ax.set_xlabel("Subjects Missing Data")
    ax.grid(axis="x", alpha=0.3)


def plot_task_count_dist(task_count_df, ax):
    ax.bar(task_count_df["Task_Count"].astype(int), task_count_df["Subject_Count"], color="#55A868")
    ax.set_title("Subjects by Task Count")
    ax.set_xlabel("Tasks with Data")
    ax.set_ylabel("Subject Count")
    ax.grid(axis="y", alpha=0.3)


def plot_age_dist(age_df, ax):
    age_df = age_df.copy()
    age_df["Age_Floor"] = age_df["Age"].astype(float).apply(lambda val: int(val))
    binned = (
        age_df.groupby("Age_Floor")["Subject_Count"]
        .sum()
        .reset_index()
        .sort_values("Age_Floor")
    )
    ax.bar(binned["Age_Floor"].astype(str), binned["Subject_Count"], color="#C44E52")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Subject Count")
    ax.tick_params(axis="x")
    ax.grid(axis="y", alpha=0.3)


def plot_group_dist(group_df, ax):
    group_df = group_df.copy()
    group_df["Group"] = group_df["Group"].astype(str)
    ax.bar(group_df["Group"], group_df["Subject_Count"], color="#8172B2")
    ax.set_title("Group Distribution")
    ax.set_ylabel("Subject Count")
    ax.tick_params(axis="x")
    ax.grid(axis="y", alpha=0.3)


def plot_sex_dist(sex_df, ax):
    sex_df = sex_df.copy()
    sex_df["Sex"] = sex_df["Sex"].astype(str).replace({"1": "Male", "2": "Female"})
    ax.bar(sex_df["Sex"], sex_df["Subject_Count"], color="#64B5CD")
    ax.set_title("Sex Distribution")
    ax.set_ylabel("Subject Count")
    ax.tick_params(axis="x")
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot task coverage summaries from EFNY_behavioral_data.csv outputs."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/EFNY/results/behavior_data/task_analysis",
        help="Output directory containing summary CSVs.",
    )
    parser.add_argument(
        "--fig_dir",
        default="outputs/EFNY/figures/behavior_data/task_analysis",
        help="Directory to save figures.",
    )

    args = parser.parse_args()

    task_missing_path = os.path.join(args.output_dir, "task_missing_counts.csv")
    task_count_path = os.path.join(args.output_dir, "task_count_distribution.csv")
    age_dist_path = os.path.join(args.output_dir, "age_distribution.csv")
    sex_dist_path = os.path.join(args.output_dir, "sex_distribution.csv")
    group_dist_path = os.path.join(args.output_dir, "group_distribution.csv")

    for path in [task_missing_path, task_count_path, age_dist_path, sex_dist_path, group_dist_path]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing summary file: {path}")

    task_missing_df = pd.read_csv(task_missing_path)
    task_count_df = pd.read_csv(task_count_path)
    age_df = pd.read_csv(age_dist_path)
    sex_df = pd.read_csv(sex_dist_path)
    group_df = pd.read_csv(group_dist_path)

    os.makedirs(args.fig_dir, exist_ok=True)
    apply_plot_style()

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    plot_task_missing(task_missing_df, axes[0, 0])
    plot_task_count_dist(task_count_df, axes[0, 1])
    plot_age_dist(age_df, axes[1, 0])
    plot_group_dist(group_df, axes[1, 1])
    plot_sex_dist(sex_df, axes[2, 0])
    axes[2, 1].axis("off")

    fig.tight_layout()
    fig_path = os.path.join(args.fig_dir, "task_analysis_overview.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to: {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
