#!/bin/bash
#SBATCH --job-name=efny_adaptive_pls_analysis      # jobname
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/task_adaptive_pls/efny_adaptive_pls_%A_%a.out   # standard output file
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/task_adaptive_pls/efny_adaptive_pls_%A_%a.err    # standard error file   
#SBATCH --partition=q_fat_c               # partition name
#SBATCH --cpus-per-task=1                 # number of cpus per task
#SBATCH --array=0-1000                    # array job, 0 is real data, 1-1000 are permutation tests

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF"

# 基础参数 - Adaptive-PLS配置
MODEL_TYPE="adaptive_pls"                 # 自适应PLS模型
N_COMPONENTS=10                            # 最大搜索范围（1-10个成分）
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
mkdir -p ${project_dir}/results/models/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE}
mkdir -p ${project_dir}/log/task_${MODEL_TYPE}

# 运行分析
echo "Starting task $SLURM_ARRAY_TASK_ID at $(date)"
echo "Task type: $TASK_TYPE"
echo "Model: $MODEL_TYPE (Adaptive)"
echo "Component search range: 1-${N_COMPONENTS}"
echo "Internal CV folds: 5"
echo "Selection criterion: canonical_correlation"

python ${project_dir}/src/scripts/run_single_task.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --model_type $MODEL_TYPE \
    --n_components $N_COMPONENTS \
    --output_dir ${project_dir}/results/models/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE} \
    --output_prefix $OUTPUT_PREFIX \
    --random_state $RANDOM_STATE \
    --log_level INFO \
    --log_file ${project_dir}/log/task_${MODEL_TYPE}/task_${SLURM_ARRAY_TASK_ID}.log

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Task $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi