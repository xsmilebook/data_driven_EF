#!/usr/bin/env python3
"""
Rename Excel sheet names for app behavioral data to consistent task names.
"""

import argparse
import glob
import os
import re

import openpyxl


def get_task_renaming_rules():
    """Return explicit sheet name renaming rules."""
    return {
        "LG": "EmotionSwitch",
        "STROOP": "ColorStroop",
        "SpatialNBack": "Spatial2Back",
        "Emotion2Backformal": "Emotion2Back",
        "Number2Backformal": "Number2Back",
        "Spatial1Backformal": "Spatial1Back",
        "EmotionStoop": "EmotionStroop",
    }


def normalize_sheet_name(sheet_name):
    """Normalize sheet name by applying explicit mapping and stripping 'formal'."""
    renaming_rules = get_task_renaming_rules()
    if sheet_name in renaming_rules:
        return renaming_rules[sheet_name]

    if "formal" in sheet_name.lower():
        return re.sub("formal", "", sheet_name, flags=re.IGNORECASE).strip()

    return sheet_name


def rename_sheets_in_excel(file_path, dry_run=False):
    """
    Rename sheets in a single Excel file.

    Returns:
        tuple[list[str], list[str]]: renamed sheets, skipped renames due to conflicts.
    """
    renamed_sheets = []
    skipped_conflicts = []

    try:
        workbook = openpyxl.load_workbook(file_path)
        original_names = list(workbook.sheetnames)

        for old_name in original_names:
            new_name = normalize_sheet_name(old_name)
            if new_name == old_name:
                continue
            if new_name in workbook.sheetnames:
                skipped_conflicts.append(f"'{old_name}' -> '{new_name}' (conflict)")
                continue

            workbook[old_name].title = new_name
            renamed_sheets.append(f"'{old_name}' -> '{new_name}'")

        if renamed_sheets and not dry_run:
            workbook.save(file_path)
        workbook.close()

        return renamed_sheets, skipped_conflicts
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return [], []


def process_directory(directory_path, pattern="*.xlsx", dry_run=False):
    """Process all Excel files in the given directory."""
    excel_files = glob.glob(os.path.join(directory_path, pattern))
    if not excel_files:
        print(f"No Excel files found in {directory_path} with pattern {pattern}")
        return

    print(f"Found {len(excel_files)} Excel files to process")

    total_renamed_files = 0
    total_renamed_sheets = 0
    total_conflicts = 0

    for idx, file_path in enumerate(excel_files, start=1):
        print(f"\nProcessing file {idx}/{len(excel_files)}: {os.path.basename(file_path)}")
        renamed_sheets, conflicts = rename_sheets_in_excel(file_path, dry_run=dry_run)

        if renamed_sheets:
            total_renamed_files += 1
            total_renamed_sheets += len(renamed_sheets)
            print(f"  Renamed {len(renamed_sheets)} sheets: {', '.join(renamed_sheets)}")

        if conflicts:
            total_conflicts += len(conflicts)
            print(f"  Skipped {len(conflicts)} due to name conflicts: {', '.join(conflicts)}")

    print("\nProcessing complete!")
    print(f"Total files processed: {len(excel_files)}")
    print(f"Files with renamed sheets: {total_renamed_files}")
    print(f"Total sheets renamed: {total_renamed_sheets}")
    if total_conflicts:
        print(f"Total rename conflicts: {total_conflicts}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Excel sheet names for app behavioral data."
    )
    parser.add_argument(
        "--input_dir",
        default=r"d:\code\data_driven_EF\data\EFNY\behavior_data\cibr_app_data",
        help="Directory containing Excel files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.xlsx",
        help="Glob pattern for Excel files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print changes without saving files.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory does not exist: {args.input_dir}")
        return
    if not os.path.isdir(args.input_dir):
        print(f"Error: Path is not a directory: {args.input_dir}")
        return

    print(f"Starting to process Excel files in: {args.input_dir}")
    process_directory(args.input_dir, pattern=args.pattern, dry_run=args.dry_run)
    print("Script completed successfully!")


if __name__ == "__main__":
    main()
