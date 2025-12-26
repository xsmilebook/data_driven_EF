import pandas as pd
import os
from collections import defaultdict
import glob

def analyze_excel_files(data_path):
    """
    分析指定路径下的所有Excel文件，统计每个任务(sheet)的被试数量
    
    Args:
        data_path: Excel文件所在路径
    
    Returns:
        dict: 任务名称 -> 被试数量的映射
        int: 总文件数量
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
            
            # 为每个sheet记录被试
            for sheet in sheet_names:
                task_subjects[sheet].add(subject_id)
                
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
    
    # 按被试数量排序
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    
    for task, count in sorted_tasks:
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"{task:<25} {count:>4} 被试 ({percentage:>5.1f}%)")
    
    # 计算统计信息
    counts = list(task_counts.values())
    if counts:
        print(f"\n统计摘要:")
        print(f"  平均被试数量: {sum(counts) / len(counts):.1f}")
        print(f"  最少被试数量: {min(counts)}")
        print(f"  最多被试数量: {max(counts)}")
        print(f"  标准差: {(sum((c - sum(counts)/len(counts))**2 for c in counts) / len(counts))**0.5:.1f}")
    
    return task_counts

if __name__ == "__main__":
    task_counts = main()