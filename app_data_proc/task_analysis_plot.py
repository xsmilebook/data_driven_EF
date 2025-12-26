"""
Visualize and summarize task coverage outputs.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def print_summary(task_summary_df, subject_level_df, meta):
    print("=" * 60)
    print("Task Coverage Summary")
    print("=" * 60)

    if meta:
        print(f"\nTotal files: {meta.get('total_files', 'N/A')}")
        print(f"Total tasks: {meta.get('task_count', 'N/A')}")
        print(f"Subject-level records: {meta.get('subject_level_records', 'N/A')}")
        print(
            "Subjects with all tasks: "
            f"{meta.get('subjects_with_all_tasks', 'N/A')}"
        )
        print(
            "Subjects with all tasks (excluding FZSS, ZYST): "
            f"{meta.get('subjects_with_all_tasks_excluding_fzss_zyst', 'N/A')}"
        )

    print(f"Tasks in summary: {len(task_summary_df)}")
    print(f"Total subjects (task presence sum): {task_summary_df['Subject_Count'].sum()}")

    print("\nTop tasks by subject count:")
    print(task_summary_df.head(10)[["Task", "Subject_Count", "Subject_Pct", "Task_Category"]])

    print("\nBottom tasks by subject count:")
    print(task_summary_df.tail(10)[["Task", "Subject_Count", "Subject_Pct", "Task_Category"]])

    if not subject_level_df.empty:
        row_stats = (
            subject_level_df.groupby("Task")["N_Rows"]
            .agg(["mean", "std", "min", "max"])
            .round(2)
        )
        print("\nRow count stats (per task):")
        print(row_stats.head(10))


def plot_task_summary(task_summary_df, subject_level_df, output_dir, top_n):
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: top tasks and category totals
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top_tasks = task_summary_df.head(top_n).iloc[::-1]
    axes[0, 0].barh(top_tasks["Task"], top_tasks["Subject_Count"], color="#4C72B0")
    axes[0, 0].set_title(f"Top {top_n} Tasks by Subject Count")
    axes[0, 0].set_xlabel("Subject Count")
    axes[0, 0].grid(axis="x", alpha=0.3)

    category_counts = (
        task_summary_df.groupby("Task_Category")["Subject_Count"].sum().sort_values()
    )
    axes[0, 1].barh(category_counts.index, category_counts.values, color="#55A868")
    axes[0, 1].set_title("Subjects by Task Category")
    axes[0, 1].set_xlabel("Subject Count")
    axes[0, 1].grid(axis="x", alpha=0.3)

    axes[1, 0].scatter(
        task_summary_df["Subject_Count"],
        task_summary_df["Mean_Rows"],
        alpha=0.7,
        color="#C44E52",
    )
    axes[1, 0].set_xlabel("Subject Count")
    axes[1, 0].set_ylabel("Mean Rows")
    axes[1, 0].set_title("Subject Count vs Mean Rows")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(task_summary_df["Mean_Missing_Rate"], bins=20, color="#8172B2")
    axes[1, 1].set_title("Distribution of Mean Missing Rate")
    axes[1, 1].set_xlabel("Mean Missing Rate")
    axes[1, 1].set_ylabel("Task Count")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "task_analysis_overview.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: row count distribution per task (if subject-level data available)
    if not subject_level_df.empty:
        top_tasks_list = task_summary_df.head(top_n)["Task"].tolist()
        subset = subject_level_df[subject_level_df["Task"].isin(top_tasks_list)]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(18, 8))
            data = [
                subset[subset["Task"] == task]["N_Rows"].values
                for task in top_tasks_list
            ]
            ax.boxplot(data, labels=top_tasks_list, showfliers=False)
            ax.set_title(f"Row Count Distribution (Top {top_n} Tasks)")
            ax.set_ylabel("Rows per Subject")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

            fig.tight_layout()
            fig_path = os.path.join(output_dir, "task_row_count_boxplot.png")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and visualize task analysis outputs."
    )
    parser.add_argument(
        "--summary_csv",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis\task_subject_counts.csv",
        help="CSV produced by task_analysis_run.py.",
    )
    parser.add_argument(
        "--subject_csv",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis\task_subject_level_summary.csv",
        help="Subject-level summary CSV.",
    )
    parser.add_argument(
        "--meta_json",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis\task_analysis_meta.json",
        help="Metadata JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis\figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Top N tasks to show in plots.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.summary_csv):
        print(f"Error: Summary CSV not found: {args.summary_csv}")
        return 1

    task_summary_df = pd.read_csv(args.summary_csv)
    if task_summary_df.empty:
        print("Error: Summary CSV is empty.")
        return 1

    subject_level_df = pd.DataFrame()
    if os.path.exists(args.subject_csv):
        subject_level_df = pd.read_csv(args.subject_csv)

    meta = None
    if os.path.exists(args.meta_json):
        meta = pd.read_json(args.meta_json, typ="series").to_dict()

    print_summary(task_summary_df, subject_level_df, meta)
    plot_task_summary(task_summary_df, subject_level_df, args.output_dir, args.top_n)

    print("\nAnalysis complete.")
    print(f"Figures saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
