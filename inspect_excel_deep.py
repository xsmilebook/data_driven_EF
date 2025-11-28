import pandas as pd
import os

file_path = r"d:\code\data_driven_EF\data\EFNY\behavior_data\cibr_app_data\THU_20231014_131_ZXM_赵夕萌_GameData.xlsx"

try:
    xl = pd.ExcelFile(file_path)
    
    print("Sheet names:", xl.sheet_names)
    
    # EmotionStroop
    sheet_name = 'EmotionStroop'
    if sheet_name not in xl.sheet_names and 'EmotionStoop' in xl.sheet_names:
        sheet_name = 'EmotionStoop'
        
    if sheet_name in xl.sheet_names:
        print(f"\n=== {sheet_name} Analysis ===")
        df = xl.parse(sheet_name)
        print("Columns:", df.columns.tolist())
        
        # Check task column
        if '任务' in df.columns:
            print("Task column unique values:", df['任务'].unique())
        elif 'task' in df.columns:
            print("Task column unique values:", df['task'].unique())
            
        if '正式阶段刺激图片/Item名' in df.columns:
             print("\nFirst 10 rows:")
             print(df[['正式阶段刺激图片/Item名', '正式阶段正确答案']].head(10))
             print("\nUnique Items:", df['正式阶段刺激图片/Item名'].unique())
             print("\nUnique Answers:", df['正式阶段正确答案'].unique())
        else:
             print("Item column not found. Head:")
             print(df.head())

    # EmotionSwitch
    if 'EmotionSwitch' in xl.sheet_names:
        print("\n=== EmotionSwitch Analysis ===")
        df = xl.parse('EmotionSwitch')
        print("Unique Items:", df['正式阶段刺激图片/Item名'].unique())
        print("Unique Answers:", df['正式阶段正确答案'].unique())

    # DT
    if 'DT' in xl.sheet_names:
        print("\n=== DT Analysis ===")
        df = xl.parse('DT')
        print("Unique Answers:", df['正式阶段正确答案'].unique())
        print(df[['正式阶段刺激图片/Item名', '正式阶段正确答案']].head(10))
        
except Exception as e:
    print(e)
