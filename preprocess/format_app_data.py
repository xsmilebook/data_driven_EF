#!/usr/bin/env python3
"""
Script to rename 'EmotionStoop' sheet to 'EmotionStroop' in Excel files.
This script processes all .xlsx files in the specified directory.
"""

import os
import pandas as pd
import openpyxl
from pathlib import Path
import glob


def rename_sheet_in_excel(file_path):
    """
    Rename 'EmotionStoop' sheet to 'EmotionStroop' in an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        bool: True if sheet was renamed, False otherwise
    """
    try:
        # Load the workbook
        workbook = openpyxl.load_workbook(file_path)
        
        # Check if 'EmotionStoop' sheet exists
        if 'EmotionStoop' in workbook.sheetnames:
            # Get the sheet
            sheet = workbook['EmotionStoop']
            # Rename the sheet
            sheet.title = 'EmotionStroop'
            # Save the workbook
            workbook.save(file_path)
            workbook.close()
            print(f"Renamed 'EmotionStoop' to 'EmotionStroop' in: {file_path}")
            return True
        else:
            workbook.close()
            print(f"No 'EmotionStoop' sheet found in: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def process_directory(directory_path):
    """
    Process all Excel files in the given directory.
    
    Args:
        directory_path (str): Path to the directory containing Excel files
    """
    # Get all .xlsx files in the directory
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    if not excel_files:
        print(f"No Excel files found in {directory_path}")
        return
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    renamed_count = 0
    processed_count = 0
    
    for file_path in excel_files:
        processed_count += 1
        print(f"\nProcessing file {processed_count}/{len(excel_files)}: {os.path.basename(file_path)}")
        
        if rename_sheet_in_excel(file_path):
            renamed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files with 'EmotionStoop' sheet renamed: {renamed_count}")


def main():
    """Main function to run the script."""
    # Define the directory path
    directory_path = r"d:\code\data_driven_EF\data\EFNY\behavior_data\cibr_app_data"
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory does not exist: {directory_path}")
        return
    
    if not os.path.isdir(directory_path):
        print(f"Error: Path is not a directory: {directory_path}")
        return
    
    print(f"Starting to process Excel files in: {directory_path}")
    process_directory(directory_path)
    print("Script completed successfully!")


if __name__ == "__main__":
    main()