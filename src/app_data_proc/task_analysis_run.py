#!/usr/bin/env python3
"""
Summarize task coverage using EFNY_behavioral_data.csv.
"""

import argparse
import json
import logging
import os

import pandas as pd


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


def derive_group_label(group_value, labels):
    """Map group string into combined label or TD."""
    if pd.isna(group_value):
        return "TD"

    group_str = str(group_value).upper()
    hits = [label for label in labels if label in group_str]
    if not hits:
        return "TD"
    return "+".join(hits)


def summarize_task_coverage(df, task_prefixes):
    """Compute per-task coverage and per-subject task counts."""
    task_columns = {}
    for prefix in task_prefixes:
        cols = [col for col in df.columns if col.startswith(f"{prefix}_")]
        task_columns[prefix] = cols

    task_rows = []
    task_presence = {}
    total_subjects = len(df)

    for prefix, cols in task_columns.items():
        if not cols:
            task_rows.append(
                {
                    "Task": prefix,
                    "Columns": 0,
                    "Subjects_With_Data": 0,
                    "Subjects_Missing_Data": total_subjects,
                    "Missing_Pct": 100.0,
                }
            )
            task_presence[prefix] = pd.Series([False] * total_subjects, index=df.index)
            continue

        has_data = df[cols].notna().any(axis=1)
        present_count = int(has_data.sum())
        missing_count = total_subjects - present_count
        task_rows.append(
            {
                "Task": prefix,
                "Columns": len(cols),
                "Subjects_With_Data": present_count,
                "Subjects_Missing_Data": missing_count,
                "Missing_Pct": round(missing_count / total_subjects * 100, 2),
            }
        )
        task_presence[prefix] = has_data

    task_missing_df = pd.DataFrame(task_rows).sort_values(
        "Subjects_Missing_Data", ascending=False
    )

    task_count_series = pd.DataFrame(task_presence).sum(axis=1)
    task_count_dist = (
        task_count_series.value_counts()
        .sort_index()
        .rename_axis("Task_Count")
        .reset_index(name="Subject_Count")
    )
    task_count_dist["Subject_Pct"] = (
        task_count_dist["Subject_Count"] / total_subjects * 100
    ).round(2)

    return task_missing_df, task_count_dist, task_columns


def main():
    parser = argparse.ArgumentParser(
        description="Summarize task coverage using EFNY_behavioral_data.csv."
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
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis",
        help="Output directory for summaries.",
    )
    parser.add_argument(
        "--log",
        default=r"d:\code\data_driven_EF\data\EFNY\log\behavior_data\task_analysis_run.log",
        help="Log file path.",
    )

    args = parser.parse_args()
    setup_logging(args.log)

    if not os.path.exists(args.behavioral_csv):
        logging.error("Behavioral CSV not found: %s", args.behavioral_csv)
        return 1
    if not os.path.exists(args.config):
        logging.error("Config JSON not found: %s", args.config)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    task_prefixes = load_task_prefixes(args.config)
    df = pd.read_csv(args.behavioral_csv)

    if "id" not in df.columns:
        logging.error("Behavioral CSV missing 'id' column.")
        return 1

    total_subjects = len(df)
    logging.info("Behavioral rows: %d", total_subjects)
    logging.info("Task prefixes: %s", ", ".join(task_prefixes))

    task_missing_df, task_count_dist, task_columns = summarize_task_coverage(df, task_prefixes)

    age_series = df["age"] if "age" in df.columns else pd.Series(dtype=float)
    sex_series = df["sex"] if "sex" in df.columns else pd.Series(dtype=object)
    group_series = df["group"] if "group" in df.columns else pd.Series(dtype=object)

    age_dist = (
        age_series.dropna()
        .round(1)
        .value_counts()
        .sort_index()
        .rename_axis("Age")
        .reset_index(name="Subject_Count")
    )
    sex_dist = (
        sex_series.fillna("NA")
        .astype(str)
        .value_counts()
        .rename_axis("Sex")
        .reset_index(name="Subject_Count")
    )

    group_labels = group_series.apply(lambda val: derive_group_label(val, ["ADHD", "DDC", "DD"]))
    group_dist = (
        group_labels.value_counts()
        .rename_axis("Group")
        .reset_index(name="Subject_Count")
    )

    age_summary = {
        "count": int(age_series.notna().sum()),
        "mean": round(float(age_series.mean()), 3) if age_series.notna().any() else None,
        "std": round(float(age_series.std(ddof=0)), 3) if age_series.notna().any() else None,
        "min": round(float(age_series.min()), 3) if age_series.notna().any() else None,
        "median": round(float(age_series.median()), 3) if age_series.notna().any() else None,
        "max": round(float(age_series.max()), 3) if age_series.notna().any() else None,
    }

    task_missing_path = os.path.join(args.output_dir, "task_missing_counts.csv")
    task_count_path = os.path.join(args.output_dir, "task_count_distribution.csv")
    age_dist_path = os.path.join(args.output_dir, "age_distribution.csv")
    sex_dist_path = os.path.join(args.output_dir, "sex_distribution.csv")
    group_dist_path = os.path.join(args.output_dir, "group_distribution.csv")
    meta_path = os.path.join(args.output_dir, "task_analysis_meta.json")

    task_missing_df.to_csv(task_missing_path, index=False, encoding="utf-8")
    task_count_dist.to_csv(task_count_path, index=False, encoding="utf-8")
    age_dist.to_csv(age_dist_path, index=False, encoding="utf-8")
    sex_dist.to_csv(sex_dist_path, index=False, encoding="utf-8")
    group_dist.to_csv(group_dist_path, index=False, encoding="utf-8")

    meta = {
        "behavioral_csv": args.behavioral_csv,
        "task_prefixes": task_prefixes,
        "total_subjects": total_subjects,
        "age_summary": age_summary,
        "task_columns": {task: cols for task, cols in task_columns.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    logging.info("Missing counts saved to: %s", task_missing_path)
    logging.info("Task count distribution saved to: %s", task_count_path)
    logging.info("Age distribution saved to: %s", age_dist_path)
    logging.info("Sex distribution saved to: %s", sex_dist_path)
    logging.info("Group distribution saved to: %s", group_dist_path)
    logging.info("Metadata saved to: %s", meta_path)

    logging.info("Age summary: %s", age_summary)
    logging.info("Group distribution: %s", dict(group_dist.values.tolist()))
    logging.info("Sex distribution: %s", dict(sex_dist.values.tolist()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
