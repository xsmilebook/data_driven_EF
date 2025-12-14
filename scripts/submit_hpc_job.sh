#!/bin/bash
#SBATCH --job-name=efny_pls_analysis      # 作业名称
#SBATCH --output=logs/efny_pls_%A_%a.out   # 输出文件
#SBATCH --error=logs/efny_pls_%A_%a.err    # 错误文件
#SBATCH --time=2:00:00                    # 运行时间限制
#SBATCH --mem=8G                          # 内存需求
#SBATCH --cpus-per-task=4                 # CPU核心数
#SBATCH --array=0-1000                    # 数组作业，0是真实数据，1-1000是置换检验

# 加载必要的模块
module load python/3.8
module load gcc/9.3.0

# 设置环境变量
export PYTHONPATH="/path/to/your/project:$PYTHONPATH"

# 激活虚拟环境
source /path/to/your/venv/bin/activate

# 基础参数
MODEL_TYPE="pls"
N_COMPONENTS=5
RANDOM_STATE=42

# 根据任务ID设置参数
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # 真实数据分析
    TASK_TYPE="real"
    OUTPUT_PREFIX="efny_real_${MODEL_TYPE}"
else
    # 置换检验
    TASK_TYPE="permutation"
    OUTPUT_PREFIX="efny_perm_${MODEL_TYPE}"
fi

# 创建输出目录
mkdir -p results/models/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE}
mkdir -p logs

# 运行分析
echo "Starting task $SLURM_ARRAY_TASK_ID at $(date)"
echo "Task type: $TASK_TYPE"
echo "Model: $MODEL_TYPE"
echo "Components: $N_COMPONENTS"

python /path/to/your/project/src/scripts/run_single_task.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --model_type $MODEL_TYPE \
    --n_components $N_COMPONENTS \
    --output_dir results/models/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE} \
    --output_prefix $OUTPUT_PREFIX \
    --random_state $RANDOM_STATE \
    --log_level INFO \
    --log_file logs/task_${SLURM_ARRAY_TASK_ID}.log

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi