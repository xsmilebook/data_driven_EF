#!/usr/bin/env python3
"""
Preprocessing script for EFNY demo data.
Transforms the original CSV file into a standardized format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
import os

def setup_logging(log_file='preprocessing.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_date(date_str):
    """Parse date string in format YYYY/MM/DD."""
    try:
        return datetime.strptime(date_str, '%Y/%m/%d')
    except (ValueError, TypeError):
        return None

def calculate_age_in_years(test_date, birth_date):
    """Calculate age in years between test date and birth date."""
    if not test_date or not birth_date:
        return None
    
    delta = test_date - birth_date
    return delta.days / 365.25  # Account for leap years

def convert_sex(sex_str):
    """Convert sex string to numeric code."""
    if pd.isna(sex_str):
        return None
    elif sex_str == '男':
        return 1
    elif sex_str == '女':
        return 2
    else:
        logging.warning(f"Unknown sex value: {sex_str}")
        return None

def merge_with_rsfmri_qc(demo_df, qc_file, output_file):
    """
    Merge demo data with rsfMRI QC data and identify missing subjects.
    Only adds the meanFD column from QC data.
    
    Args:
        demo_df: Demo dataframe with subid column
        qc_file: Path to rsfMRI QC CSV file
        output_file: Path to output merged CSV file
    
    Returns:
        merged_df: Merged dataframe with meanFD column added
        missing_subjects: List of subjects in QC but not in demo
    """
    logging.info(f"Starting merge with rsfMRI QC data from {qc_file}")
    
    # Read the QC data
    try:
        qc_df = pd.read_csv(qc_file)
        logging.info(f"Successfully loaded {len(qc_df)} rows from {qc_file}")
    except Exception as e:
        logging.error(f"Failed to read QC file: {e}")
        return None, []
    
    # Check if meanFD column exists in QC data
    if 'meanFD' not in qc_df.columns:
        logging.error("QC file does not contain 'meanFD' column")
        return None, []
    
    # Get unique subjects from both datasets
    demo_subjects = set(demo_df['subid'].dropna().astype(str))
    qc_subjects = set(qc_df['subid'].dropna().astype(str))
    
    # Find subjects that are in QC but not in demo
    missing_subjects = list(qc_subjects - demo_subjects)
    missing_subjects.sort()
    
    # Find subjects that are in both datasets (intersection)
    common_subjects = demo_subjects & qc_subjects
    
    logging.info(f"Demo subjects: {len(demo_subjects)}")
    logging.info(f"QC subjects: {len(qc_subjects)}")
    logging.info(f"Common subjects: {len(common_subjects)}")
    logging.info(f"Missing from demo: {len(missing_subjects)}")
    
    if missing_subjects:
        logging.warning(f"Subjects in QC but missing from demo data:")
        for subject in missing_subjects[:10]:  # Show first 10
            logging.warning(f"  - {subject}")
        if len(missing_subjects) > 10:
            logging.warning(f"  ... and {len(missing_subjects) - 10} more")
    
    # Create a copy of demo_df to avoid modifying the original
    merged_df = demo_df.copy()
    
    # Only keep the meanFD column from QC data, using left join to preserve all demo subjects
    meanfd_df = qc_df[['subid', 'meanFD']].copy()
    merged_df = pd.merge(merged_df, meanfd_df, on='subid', how='inner')
    
    logging.info(f"Successfully added meanFD column to {len(merged_df)} subjects")

    merged_df = merged_df[merged_df['age'] < 26]
    merged_df = merged_df[merged_df['group'] == ""]
    
    # Save merged data
    try:
        merged_df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved merged data to {output_file}")
        
        # Print merge summary
        logging.info("\n=== Merge Summary ===")
        logging.info(f"Total subjects in output: {len(merged_df)}")
        logging.info(f"Subjects with meanFD data: {merged_df['meanFD'].notna().sum()}")
        logging.info(f"Demo only subjects: {len(demo_subjects - qc_subjects)}")
        logging.info(f"QC only subjects: {len(missing_subjects)}")
        
        return merged_df, missing_subjects
        
    except Exception as e:
        logging.error(f"Failed to save merged file: {e}")
        return None, missing_subjects

def preprocess_efny_data(input_file, output_file, qc_file=None, merged_output_file=None):
    """
    Preprocess EFNY demo data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        qc_file: Optional path to rsfMRI QC CSV file for merging
        merged_output_file: Optional path to merged output CSV file
    """
    logging.info(f"Starting preprocessing of {input_file}")
    
    # Read the input CSV file
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return False
    
    # Create output dataframe
    output_df = pd.DataFrame()
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Extract basic information
            id_value = row['ID']
            subid = row['MRI_ID']
            if pd.isna(id_value) and not pd.isna(subid):
                id_value = subid.split("_")[2]

            # Calculate age from test day and birth day
            test_day = parse_date(str(row['Test_day']))
            birth_day = parse_date(str(row['Birth_day']))
            calculated_age = calculate_age_in_years(test_day, birth_day)
            original_age = row['Age']
            
            # Age validation and warning
            if calculated_age is not None and original_age is not None:
                age_diff = abs(calculated_age - original_age)
                if age_diff > 1:
                    logging.warning(f"Row {idx}: Large age difference detected - "
                                  f"Calculated: {calculated_age:.2f}, Original: {original_age:.2f}, "
                                  f"Difference: {age_diff:.2f}")
            
            # Use calculated age if available, otherwise use original age
            age = calculated_age if calculated_age is not None else original_age
            
            # Convert sex
            sex = convert_sex(row['Sex'])
            
            # Extract group information
            group = str(row['Group（DDC：dyscalculia，DD：developmental dyslexia）'])
            sub_group = ""
            group_cate_list = ["ADHD", "DDC", "DD"]
            for group_cate in group_cate_list:
                if group_cate in group:
                    sub_group += group_cate + ","
            
            # Create processed row
            processed_row = {
                'id': int(id_value),
                'subid': subid,
                'age': age,
                'sex': sex,
                'group': sub_group.strip(','),
                'Raven_Score': row['Raven_Score'],
                'Hand': row['Hand'],
                'Parents_education': row['Parents_education'],
                'Parents': row['Parents']
            }
            
            # Add to output dataframe
            output_df = pd.concat([output_df, pd.DataFrame([processed_row])], ignore_index=True)
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            continue
    
    # Save the processed data
    try:
        output_df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved {len(output_df)} rows to {output_file}")
        
        # Print summary statistics
        logging.info("\n=== Processing Summary ===")
        logging.info(f"Total rows processed: {len(output_df)}")
        logging.info(f"Valid ages: {output_df['age'].notna().sum()}")
        logging.info(f"Valid sex codes: {output_df['sex'].notna().sum()}")
        logging.info(f"Valid groups: {output_df['group'].notna().sum()}")
        
        # Show age statistics
        valid_ages = output_df['age'].dropna()
        if len(valid_ages) > 0:
            logging.info(f"Age range: {valid_ages.min():.2f} - {valid_ages.max():.2f} years")
            logging.info(f"Mean age: {valid_ages.mean():.2f} years")
        
        # If QC file is provided, perform merge
        if qc_file and merged_output_file:
            merged_df, missing_subjects = merge_with_rsfmri_qc(output_df, qc_file, merged_output_file)
            return True, missing_subjects
        
        return True, []
        
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")
        return False, []

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Preprocess EFNY demo data')
    parser.add_argument('--input', '-i', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_demo_all.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_demo_processed.csv',
                       help='Output CSV file path')
    parser.add_argument('--qc-file', '-q',
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\qc\\rest_fd_summary.csv',
                       help='rsfMRI QC CSV file path for merging')
    parser.add_argument('--merged-output', '-m',
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_demo_with_rsfmri.csv',
                       help='Merged output CSV file path')
    parser.add_argument('--no-merge', action='store_true',
                       help='Skip merging with QC data')
    parser.add_argument('--log', '-l', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\log\\preprocess\\preprocess_efny_demo.log',
                       help='Log file path')
    
    args = parser.parse_args()
    
    setup_logging(args.log)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logging.error(f"Input file does not exist: {args.input}")
        return 1
    
    # Run preprocessing
    if args.no_merge:
        success, missing_subjects = preprocess_efny_data(args.input, args.output)
    else:
        # Check if QC file exists
        if not os.path.exists(args.qc_file):
            logging.error(f"QC file does not exist: {args.qc_file}")
            return 1
        
        success, missing_subjects = preprocess_efny_data(
            args.input, args.output, args.qc_file, args.merged_output
        )
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())