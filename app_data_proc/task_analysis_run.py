"""
Analyze task coverage in app behavioral Excel files and generate summary outputs.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd


def get_task_renaming_rules():
    """Return explicit sheet name normalization rules."""
    return {
        "LG": "EmotionSwitch",
        "STROOP": "ColorStroop",
        "SpatialNBack": "Spatial2Back",
        "Emotion2Backformal": "Emotion2Back",
        "Number2Backformal": "Number2Back",
        "Spatial1Backformal": "Spatial1Back",
        "EmotionStoop": "EmotionStroop",
    }


def normalize_task_name(task_name):
    """Normalize task names to canonical labels."""
    mapping = get_task_renaming_rules()
    if task_name in mapping:
        return mapping[task_name]
    if "formal" in task_name.lower():
        return re.sub("formal", "", task_name, flags=re.IGNORECASE).strip()
    return task_name


def categorize_task(task_name):
    """Assign a coarse task category."""
    task_lower = task_name.lower()
    if "back" in task_lower:
        return "NBack"
    if "stroop" in task_lower:
        return "Stroop"
    if "flanker" in task_lower:
        return "Flanker"
    if "sst" in task_lower or "stop" in task_lower:
        return "StopSignal"
    if "dccs" in task_lower or "switch" in task_lower:
        return "TaskSwitch"
    if "dt" in task_lower:
        return "Decision"
    if "cpt" in task_lower:
        return "SustainedAttention"
    if "gng" in task_lower:
        return "GoNoGo"
    if "kt" in task_lower:
        return "OtherCognitive"
    if "fzss" in task_lower or "zyst" in task_lower:
        return "OtherCognitive"
    return "Other"


def extract_subject_id(file_name):
    """Extract subject id from the filename."""
    base = os.path.basename(file_name)
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[2]
    return base.replace("_GameData.xlsx", "").replace(".xlsx", "")


def analyze_excel_files(data_path, pattern):
    """Analyze Excel files and return summary records."""
    excel_files = glob.glob(os.path.join(data_path, pattern))
    print(f"Found {len(excel_files)} Excel files")

    task_subjects = defaultdict(set)
    task_column_counts = defaultdict(Counter)
    subject_level_records = []
    file_errors = []

    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        subject_id = extract_subject_id(file_name)

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
        except Exception as exc:
            file_errors.append((file_name, str(exc)))
            print(f"Failed to read {file_name}: {exc}")
            continue

        for sheet in sheet_names:
            normalized_sheet = normalize_task_name(sheet)
            task_subjects[normalized_sheet].add(subject_id)

            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
            except Exception as exc:
                file_errors.append((file_name, f"{sheet}: {exc}"))
                print(f"Failed to read sheet {sheet} in {file_name}: {exc}")
                continue

            n_rows, n_cols = df.shape
            total_cells = n_rows * n_cols
            missing_cells = int(df.isna().sum().sum())
            missing_rate = missing_cells / total_cells if total_cells else 0.0

            numeric_cols = df.select_dtypes(include="number").columns
            numeric_cols_count = len(numeric_cols)
            if n_rows and numeric_cols_count:
                numeric_missing_rate = (
                    df[numeric_cols].isna().sum().sum() / (n_rows * numeric_cols_count)
                )
            else:
                numeric_missing_rate = None

            subject_level_records.append(
                {
                    "Subject_ID": subject_id,
                    "Task": normalized_sheet,
                    "Task_Category": categorize_task(normalized_sheet),
                    "File_Name": file_name,
                    "Sheet_Name": sheet,
                    "N_Rows": n_rows,
                    "N_Cols": n_cols,
                    "Missing_Cells": missing_cells,
                    "Missing_Rate": round(missing_rate, 6),
                    "Numeric_Cols": numeric_cols_count,
                    "Numeric_Missing_Rate": (
                        None if numeric_missing_rate is None else round(numeric_missing_rate, 6)
                    ),
                }
            )

            for col in df.columns:
                task_column_counts[normalized_sheet][col] += 1

    task_counts = {task: len(subjects) for task, subjects in task_subjects.items()}

    return task_counts, task_subjects, subject_level_records, task_column_counts, len(excel_files), file_errors


def count_subjects_with_all_tasks(task_subjects, excluded_tasks=None):
    """Count subjects who have all tasks, optionally excluding some tasks."""
    excluded = set(excluded_tasks or [])
    task_sets = [subjects for task, subjects in task_subjects.items() if task not in excluded]

    if not task_sets:
        return 0

    common_subjects = set(task_sets[0])
    for subject_set in task_sets[1:]:
        common_subjects &= subject_set

    return len(common_subjects)


def build_task_summary(task_subjects, subject_level_df, total_files):
    """Aggregate per-task statistics."""
    summary_rows = []

    for task, subjects in task_subjects.items():
        task_df = subject_level_df[subject_level_df["Task"] == task]
        row_counts = task_df["N_Rows"].tolist()
        col_counts = task_df["N_Cols"].tolist()
        missing_rates = task_df["Missing_Rate"].tolist()
        numeric_cols = task_df["Numeric_Cols"].tolist()
        numeric_missing = task_df["Numeric_Missing_Rate"].dropna().tolist()

        summary_rows.append(
            {
                "Task": task,
                "Task_Category": categorize_task(task),
                "Subject_Count": len(subjects),
                "Subject_Pct": round(len(subjects) / total_files * 100, 2)
                if total_files
                else 0.0,
                "Record_Count": len(task_df),
                "Total_Rows": int(sum(row_counts)),
                "Mean_Rows": round(sum(row_counts) / len(row_counts), 2) if row_counts else 0.0,
                "Median_Rows": round(pd.Series(row_counts).median(), 2)
                if row_counts
                else 0.0,
                "Min_Rows": int(min(row_counts)) if row_counts else 0,
                "Max_Rows": int(max(row_counts)) if row_counts else 0,
                "Std_Rows": round(pd.Series(row_counts).std(ddof=0), 2)
                if row_counts
                else 0.0,
                "Mean_Cols": round(sum(col_counts) / len(col_counts), 2)
                if col_counts
                else 0.0,
                "Mean_Missing_Rate": round(sum(missing_rates) / len(missing_rates), 6)
                if missing_rates
                else 0.0,
                "Median_Missing_Rate": round(pd.Series(missing_rates).median(), 6)
                if missing_rates
                else 0.0,
                "Max_Missing_Rate": round(max(missing_rates), 6) if missing_rates else 0.0,
                "Mean_Numeric_Cols": round(sum(numeric_cols) / len(numeric_cols), 2)
                if numeric_cols
                else 0.0,
                "Mean_Numeric_Missing_Rate": round(
                    sum(numeric_missing) / len(numeric_missing), 6
                )
                if numeric_missing
                else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Subject_Count", ascending=False)
    return summary_df


def build_column_coverage(task_subjects, task_column_counts, max_columns_per_task):
    """Build column coverage summary."""
    rows = []

    for task, col_counts in task_column_counts.items():
        subject_count = len(task_subjects[task])
        for col, count in col_counts.most_common():
            rows.append(
                {
                    "Task": task,
                    "Column": col,
                    "Subjects_With_Column": count,
                    "Subject_Pct": round(count / subject_count * 100, 2)
                    if subject_count
                    else 0.0,
                }
            )

    coverage_df = pd.DataFrame(rows)
    if coverage_df.empty:
        return coverage_df

    coverage_df = coverage_df.sort_values(
        ["Task", "Subjects_With_Column"], ascending=[True, False]
    )

    if max_columns_per_task and max_columns_per_task > 0:
        coverage_df = (
            coverage_df.groupby("Task")
            .head(max_columns_per_task)
            .reset_index(drop=True)
        )

    return coverage_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task coverage and generate summaries."
    )
    parser.add_argument(
        "--input_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\behavior_data\cibr_app_data",
        help="Directory containing Excel files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_GameData.xlsx",
        help="Glob pattern for Excel files.",
    )
    parser.add_argument(
        "--output_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\results\behavior_data\task_analysis",
        help="Output directory for summaries.",
    )
    parser.add_argument(
        "--max_columns_per_task",
        type=int,
        default=50,
        help="Limit column coverage output per task (0 for all).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory does not exist: {args.input_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    (
        task_counts,
        task_subjects,
        subject_level_records,
        task_column_counts,
        total_files,
        errors,
    ) = analyze_excel_files(args.input_dir, args.pattern)

    if errors:
        print(f"Warnings: {len(errors)} files/sheets failed to read.")
        for file_name, error in errors[:5]:
            print(f"  - {file_name}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    subject_level_df = pd.DataFrame(subject_level_records)
    task_summary_df = build_task_summary(task_subjects, subject_level_df, total_files)
    column_coverage_df = build_column_coverage(
        task_subjects, task_column_counts, args.max_columns_per_task
    )

    summary_path = os.path.join(args.output_dir, "task_subject_counts.csv")
    subject_level_path = os.path.join(args.output_dir, "task_subject_level_summary.csv")
    column_coverage_path = os.path.join(args.output_dir, "task_column_coverage.csv")
    meta_path = os.path.join(args.output_dir, "task_analysis_meta.json")

    task_summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    subject_level_df.to_csv(subject_level_path, index=False, encoding="utf-8")
    column_coverage_df.to_csv(column_coverage_path, index=False, encoding="utf-8")

    all_tasks_count = count_subjects_with_all_tasks(task_subjects)
    all_tasks_excluding = count_subjects_with_all_tasks(task_subjects, excluded_tasks=["FZSS", "ZYST"])

    meta = {
        "total_files": total_files,
        "task_count": len(task_counts),
        "subject_level_records": len(subject_level_records),
        "error_count": len(errors),
        "subjects_with_all_tasks": all_tasks_count,
        "subjects_with_all_tasks_excluding_fzss_zyst": all_tasks_excluding,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print("\nAnalysis complete.")
    print(f"Total files: {total_files}")
    print(f"Tasks found: {len(task_counts)}")
    print(f"Subject-level records: {len(subject_level_records)}")
    print(f"Subjects with all tasks: {all_tasks_count}")
    print(f"Subjects with all tasks (excluding FZSS, ZYST): {all_tasks_excluding}")
    print(f"Summary saved to: {summary_path}")
    print(f"Subject-level summary saved to: {subject_level_path}")
    print(f"Column coverage saved to: {column_coverage_path}")
    print(f"Metadata saved to: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
