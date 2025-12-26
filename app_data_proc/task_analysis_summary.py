"""
任务被试数量分析总结
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('task_subject_counts.csv')

print("=" * 60)
print("任务被试数量分析总结")
print("=" * 60)

print(f"\n总文件数量: 583")
print(f"总任务数量: {len(df)}")
print(f"总被试数量: {df['Subject_Count'].sum()}")

print("\n" + "=" * 40)
print("各任务类型统计:")
print("=" * 40)

category_stats = df.groupby('Task_Category').agg({
    'Subject_Count': ['count', 'sum', 'mean'],
    'Percentage': 'mean'
}).round(1)

category_stats.columns = ['任务数量', '总被试数', '平均被试数', '平均百分比']
print(category_stats)

print("\n" + "=" * 40)
print("被试数量最多的前5个任务:")
print("=" * 40)
print(df.head()[['Task', 'Subject_Count', 'Percentage', 'Task_Category']])

print("\n" + "=" * 40)
print("被试数量最少的前5个任务:")
print("=" * 40)
print(df.tail()[['Task', 'Subject_Count', 'Percentage', 'Task_Category']])

# 创建可视化
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 各任务被试数量条形图
top_15 = df.head(15)
axes[0,0].barh(range(len(top_15)), top_15['Subject_Count'])
axes[0,0].set_yticks(range(len(top_15)))
axes[0,0].set_yticklabels(top_15['Task'])
axes[0,0].set_xlabel('被试数量')
axes[0,0].set_title('前15个任务的被试数量')
axes[0,0].grid(axis='x', alpha=0.3)

# 2. 任务类型分布饼图
category_counts = df.groupby('Task_Category')['Subject_Count'].sum()
axes[0,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
axes[0,1].set_title('各任务类型被试数量分布')

# 3. 任务类型平均被试数
axes[1,0].bar(range(len(category_stats)), category_stats['平均被试数'])
axes[1,0].set_xticks(range(len(category_stats)))
axes[1,0].set_xticklabels(category_stats.index, rotation=45)
axes[1,0].set_ylabel('平均被试数量')
axes[1,0].set_title('各任务类型平均被试数量')
axes[1,0].grid(axis='y', alpha=0.3)

# 4. 被试数量分布直方图
axes[1,1].hist(df['Subject_Count'], bins=10, edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('被试数量')
axes[1,1].set_ylabel('任务数量')
axes[1,1].set_title('被试数量分布')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("分析完成！图表已保存为: task_analysis_summary.png")
print("=" * 60)