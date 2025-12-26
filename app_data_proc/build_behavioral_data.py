#!/usr/bin/env python3
"""
Script to build behavioral data table by merging EFNY metrics with demographic data.
Extracts ID from subject_code (3rd part after underscore split) and performs inner join.
"""

import pandas as pd
import numpy as np
import logging
import argparse
import os

def setup_logging(log_file='behavioral_data_processing.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def extract_id_from_subject_code(subject_code):
    """
    Extract ID from subject_code by splitting on underscore and taking the 3rd part.
    
    Args:
        subject_code: String like 'THU_20231014_131_ZXM_赵夕萌'
    
    Returns:
        int: Extracted ID (e.g., 131)
    """
    try:
        parts = subject_code.split('_')
        if len(parts) >= 3:
            return int(parts[2])  # 3rd part after split
        else:
            logging.warning(f"Invalid subject_code format: {subject_code}")
            return None
    except (ValueError, IndexError, AttributeError) as e:
        logging.error(f"Error extracting ID from subject_code '{subject_code}': {e}")
        return None

def build_behavioral_data(metrics_file, demo_file, output_file, keep_complete_only=False):
    """
    Build behavioral data table by merging metrics with demographic data.
    
    Args:
        metrics_file: Path to EFNY_metrics.csv
        demo_file: Path to EFNY_demo_with_rsfmri.csv
        output_file: Path to output merged behavioral data
        keep_complete_only: If True, only keep subjects with complete behavioral data
    
    Returns:
        bool: Success status
    """
    logging.info(f"Starting behavioral data build process")
    logging.info(f"Metrics file: {metrics_file}")
    logging.info(f"Demo file: {demo_file}")
    logging.info(f"Keep complete data only: {keep_complete_only}")
    
    # Read metrics data
    try:
        metrics_df = pd.read_csv(metrics_file, encoding="utf-8")
        logging.info(f"Successfully loaded {len(metrics_df)} rows from metrics file")
        logging.info(f"Metrics columns: {list(metrics_df.columns)}")
    except Exception as e:
        logging.error(f"Failed to read metrics file: {e}")
        return False
    
    # Read demo data
    try:
        demo_df = pd.read_csv(demo_file, encoding="utf-8")
        logging.info(f"Successfully loaded {len(demo_df)} rows from demo file")
        logging.info(f"Demo columns: {list(demo_df.columns)}")
    except Exception as e:
        logging.error(f"Failed to read demo file: {e}")
        return False
    
    # Extract ID from subject_code
    logging.info("Extracting ID from subject_code...")
    metrics_df['id'] = metrics_df['subject_code'].apply(extract_id_from_subject_code)
    
    # Check for invalid IDs
    invalid_ids = metrics_df[metrics_df['id'].isna()]
    if len(invalid_ids) > 0:
        logging.warning(f"Found {len(invalid_ids)} rows with invalid subject_code format:")
        for idx, row in invalid_ids.head(10).iterrows():
            logging.warning(f"  - Row {idx}: {row['subject_code']}")
        if len(invalid_ids) > 10:
            logging.warning(f"  ... and {len(invalid_ids) - 10} more")
    
    # Remove rows with invalid IDs
    metrics_df = metrics_df[metrics_df['id'].notna()]
    logging.info(f"Metrics data after ID extraction: {len(metrics_df)} rows")
    
    # Drop subject_code and file_name columns as requested
    columns_to_drop = ['subject_code', 'file_name']
    metrics_df = metrics_df.drop(columns=columns_to_drop)
    logging.info(f"Dropped columns: {columns_to_drop}")
    logging.info(f"Remaining metrics columns: {list(metrics_df.columns)}")
    
    # Perform inner join on ID
    logging.info("Performing inner join on ID...")
    merged_df = pd.merge(demo_df, metrics_df, on='id', how='inner')
    
    # Log merge statistics
    metrics_ids = set(metrics_df['id'].dropna().astype(int))
    demo_ids = set(demo_df['id'].dropna().astype(int))
    common_ids = metrics_ids & demo_ids
    missing_from_demo = metrics_ids - demo_ids
    missing_from_metrics = demo_ids - metrics_ids
    
    logging.info(f"Metrics IDs: {len(metrics_ids)}")
    logging.info(f"Demo IDs: {len(demo_ids)}")
    logging.info(f"Common IDs: {len(common_ids)}")
    logging.info(f"Missing from demo: {len(missing_from_demo)}")
    logging.info(f"Missing from metrics: {len(missing_from_metrics)}")
    
    if missing_from_demo:
        logging.warning(f"IDs in metrics but missing from demo data:")
        for id_val in sorted(list(missing_from_demo))[:10]:
            logging.warning(f"  - {id_val}")
        if len(missing_from_demo) > 10:
            logging.warning(f"  ... and {len(missing_from_demo) - 10} more")
    
    if missing_from_metrics:
        logging.warning(f"IDs in demo but missing from metrics data:")
        for id_val in sorted(list(missing_from_metrics))[:10]:
            logging.warning(f"  - {id_val}")
        if len(missing_from_metrics) > 10:
            logging.warning(f"  ... and {len(missing_from_metrics) - 10} more")
    
    # Filter for complete data only if requested
    if keep_complete_only:
        logging.info("Filtering for subjects with complete behavioral data...")
        
        # Identify behavioral metrics columns (exclude demo columns)
        behavioral_cols = [col for col in merged_df.columns if col not in demo_df.columns and col != 'id']
        logging.info(f"Behavioral metrics columns: {len(behavioral_cols)}")
        
        # Count missing values per subject
        missing_counts = merged_df[behavioral_cols].isnull().sum(axis=1)
        complete_subjects = missing_counts == 0
        
        logging.info(f"Subjects with complete data: {complete_subjects.sum()}/{len(merged_df)}")
        logging.info(f"Subjects with missing data: {(~complete_subjects).sum()}/{len(merged_df)}")
        
        # Show distribution of missing values
        missing_dist = missing_counts.value_counts().sort_index()
        logging.info("Missing values distribution:")
        for missing_num, count in missing_dist.items():
            if missing_num <= 5:  # Only show up to 5 missing values
                logging.info(f"  {count} subjects with {missing_num} missing values")
            elif missing_num == missing_dist.index[-1]:  # Show the maximum
                logging.info(f"  {count} subjects with {missing_num}+ missing values")
        
        # Filter to keep only complete subjects
        merged_df = merged_df[complete_subjects].copy()
        logging.info(f"Filtered dataset: {len(merged_df)} subjects with complete behavioral data")
    
    # Final merged data info
    logging.info(f"Final merged dataset: {len(merged_df)} rows")
    logging.info(f"Final columns: {list(merged_df.columns)}")
    
    # Save the merged data
    try:
        merged_df.to_csv(output_file, index=False, encoding="utf-8")
        logging.info(f"Successfully saved merged behavioral data to {output_file}")
        
        # Print summary statistics
        logging.info("\n=== Behavioral Data Build Summary ===")
        logging.info(f"Total subjects: {len(merged_df)}")
        logging.info(f"Behavioral metrics columns: {len([col for col in merged_df.columns if col not in demo_df.columns])}")
        logging.info(f"Demo columns: {len(demo_df.columns)}")
        logging.info(f"Total columns: {len(merged_df.columns)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Build behavioral data table by merging EFNY metrics with demographic data')
    parser.add_argument('--metrics', '-m', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\metrics\\EFNY_beh_metrics.csv',
                       help='Input metrics CSV file path')
    parser.add_argument('--demo', '-d', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_demo_with_rsfmri.csv',
                       help='Input demo CSV file path')
    parser.add_argument('--output', '-o', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_behavioral_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--log', '-l', 
                       default='d:\\code\\data_driven_EF\\data\\EFNY\\log\\preprocess\\build_behavioral_data.log',
                       help='Log file path')
    
    args = parser.parse_args()
    
    setup_logging(args.log)
    
    # Check if input files exist
    if not os.path.exists(args.metrics):
        logging.error(f"Metrics file does not exist: {args.metrics}")
        return 1
    
    if not os.path.exists(args.demo):
        logging.error(f"Demo file does not exist: {args.demo}")
        return 1
    
    # Run the build process
    success = build_behavioral_data(args.metrics, args.demo, args.output, keep_complete_only=False)
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
