"""
分析任务被试数量的脚本，包含任务名称标准化功能
"""
import pandas as pd
import os
from collections import defaultdict
import glob

def normalize_task_name(task_name):
    """
    标准化任务名称，处理格式不规范的sheet名称
    
    Args:
        task_name: 原始任务名称
    
    Returns:
        str: 标准化后的任务名称
    """
    # 定义任务名称映射规则
    task_mapping = {
        'LG': 'EmotionSwitch',
        'STROOP': 'ColorStroop', 
        'SpatialNBack': 'Spatial2Back'
    }
    
    # 如果任务名称在映射表中，返回对应的规范名称
    if task_name in task_mapping:
        return task_mapping[task_name]
    
    # 其他情况返回原名称
    return task_name

def categorize_task(task_name):
    """根据任务名称分类"""
    task_lower = task_name.lower()
    
    if 'back' in task_lower:
        return 'N-back任务'
    elif 'stroop' in task_lower:
        return 'Stroop任务'
    elif 'flanker' in task_lower:
        return 'Flanker任务'
    elif 'sst' in task_lower or 'stop' in task_lower:
        return 'Stop Signal任务'
    elif 'dccs' in task_lower or 'switch' in task_lower:
        return '任务切换'
    elif 'dt' in task_lower:
        return '决策任务'
    elif 'cpt' in task_lower:
        return '持续注意任务'
    elif 'gng' in task_lower:
        return 'Go/No-Go任务'
    elif 'kt' in task_lower:
        return '其他认知任务'
    elif 'fzss' in task_lower or 'zyst' in task_lower:
        return '其他认知任务'
    else:
        return '其他任务'

def analyze_excel_files(data_path):
    """
    分析指定路径下的所有Excel文件，统计每个任务(sheet)的被试数量
    
    Args:
        data_path: Excel文件所在路径
    
    Returns:
        dict: 任务名称 -> 被试数量的映射
        int: 总文件数量
        list: 文件错误列表
    """
    # 获取所有Excel文件
    excel_files = glob.glob(os.path.join(data_path, "*_GameData.xlsx"))
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    # 用于统计每个任务的被试数量
    task_subjects = defaultdict(set)  # 任务名称 -> 被试ID集合
    file_errors = []
    
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        
        # 从文件名提取被试ID (格式: THU_日期_编号_姓名_ GameData.xlsx)
        try:
            parts = file_name.split('_')
            if len(parts) >= 3:
                subject_id = parts[2]  # 提取编号作为被试ID
            else:
                subject_id = file_name.replace('_GameData.xlsx', '')
        except:
            subject_id = file_name.replace('_GameData.xlsx', '')
        
        try:
            # 读取Excel文件
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            # 为每个sheet记录被试，使用标准化后的任务名称
            for sheet in sheet_names:
                normalized_sheet = normalize_task_name(sheet)
                task_subjects[normalized_sheet].add(subject_id)
                
        except Exception as e:
            file_errors.append((file_name, str(e)))
            print(f"读取文件失败: {file_name}, 错误: {e}")
    
    # 转换统计结果为数量
    task_counts = {task: len(subjects) for task, subjects in task_subjects.items()}
    
    return task_counts, len(excel_files), file_errors

def main():
    # 设置数据路径
    data_path = r"d:\code\data_driven_EF\data\EFNY\behavior_data\cibr_app_data"
    
    print("开始分析任务被试数量...")
    print("=" * 60)
    
    # 分析数据
    task_counts, total_files, errors = analyze_excel_files(data_path)
    
    print("\n分析结果:")
    print("=" * 60)
    
    if errors:
        print(f"警告: {len(errors)} 个文件读取失败")
        for file_name, error in errors[:5]:  # 只显示前5个错误
            print(f"  - {file_name}: {error}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    
    print(f"\n总文件数量: {total_files}")
    print(f"成功分析的任务数量: {len(task_counts)}")
    print("\n各任务被试数量统计:")
    print("-" * 60)
    
    # 创建DataFrame用于更好的展示和导出
    df_tasks = pd.DataFrame(list(task_counts.items()), columns=['Task', 'Subject_Count'])
    df_tasks = df_tasks.sort_values('Subject_Count', ascending=False)
    df_tasks['Percentage'] = (df_tasks['Subject_Count'] / total_files * 100).round(1)
    df_tasks['Task_Category'] = df_tasks['Task'].apply(categorize_task)
    
    # 显示结果
    for _, row in df_tasks.iterrows():
        print(f"{row['Task']:<25} {row['Subject_Count']:>4} 被试 ({row['Percentage']:>5.1f}%)  {row['Task_Category']}")
    
    # 计算统计信息
    counts = list(task_counts.values())
    if counts:
        print(f"\n统计摘要:")
        print(f"  平均被试数量: {sum(counts) / len(counts):.1f}")
        print(f"  最少被试数量: {min(counts)}")
        print(f"  最多被试数量: {max(counts)}")
        print(f"  标准差: {(sum((c - sum(counts)/len(counts))**2 for c in counts) / len(counts))**0.5:.1f}")
    
    # 导出结果到CSV
    output_file = 'task_subject_counts_corrected.csv'
    df_tasks.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已导出到: {output_file}")
    
    return df_tasks

if __name__ == "__main__":
    df_tasks = main()