import pandas

def get_sublist_from_csv(csv_path: str) -> pandas.DataFrame:
    """Extract subid column from CSV file"""
    df = pandas.read_csv(csv_path)
    return df[['subid']]


if __name__ == "__main__":
    """Generate sublist.txt from EFNY_behavioral_data.csv"""
    sublist_path = "D:\\code\\data_driven_EF\\data\\EFNY\\table\\sublist\\sublist.txt"
    csv_path = "D:\\code\\data_driven_EF\\data\\EFNY\\table\\demo\\EFNY_behavioral_data.csv"
    
    # Read CSV and extract subid column
    df = get_sublist_from_csv(csv_path)
    
    # Write subids to file
    with open(sublist_path, "w", encoding="utf-8", newline="") as f:
        for subid in df['subid']:
            f.write(str(subid) + "\n")
        