#!/bin/bash
#SBATCH --job-name=efny_perm_adaptive_pls      # jobname
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/perm_adaptive_pls/efny_perm_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/perm_adaptive_pls/efny_perm_%A_%a.err
#SBATCH --partition=q_fat_c
#SBATCH --cpus-per-task=1
#SBATCH --array=1-1000                    # permutation tasks

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF"

MODEL_TYPE="adaptive_pls"
N_COMPONENTS=10
RANDOM_STATE=42

TASK_TYPE="permutation"

mkdir -p ${project_dir}/results/perm/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE}
mkdir -p ${project_dir}/log/perm_adaptive_pls

echo "Starting PERM task $SLURM_ARRAY_TASK_ID at $(date)"
echo "Model: $MODEL_TYPE (Adaptive->PLS fixed n_components)"
echo "Component search range (for real-data Adaptive-PLS): 1-${N_COMPONENTS}"

python ${project_dir}/src/scripts/run_single_task.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --model_type $MODEL_TYPE \
    --n_components $N_COMPONENTS \
    --output_dir ${project_dir}/results/perm/task_${SLURM_ARRAY_TASK_ID}_${MODEL_TYPE} \
    --output_prefix efny_perm_${MODEL_TYPE} \
    --random_state $RANDOM_STATE \
    --log_level INFO \
    --log_file ${project_dir}/log/perm_adaptive_pls/task_${SLURM_ARRAY_TASK_ID}.log

if [ $? -eq 0 ]; then
    echo "Permutation task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Permutation task $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi

