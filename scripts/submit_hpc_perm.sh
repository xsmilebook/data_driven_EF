#!/bin/bash
#SBATCH --job-name=efny_perm_adaptive_pls      # jobname
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --output=outputs/logs/perm_adaptive_pls/%x_%A_%a.out
#SBATCH --error=outputs/logs/perm_adaptive_pls/%x_%A_%a.err
#SBATCH --partition=q_fat_c
#SBATCH --cpus-per-task=1
#SBATCH --array=1-1000                    # permutation tasks

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
paths_config="${project_dir}/configs/paths.yaml"

MODEL_TYPE="adaptive_pls"
RANDOM_STATE=42

mkdir -p ${project_dir}/outputs/logs/perm_${MODEL_TYPE}

echo "Starting PERM task $SLURM_ARRAY_TASK_ID at $(date)"
echo "Model: $MODEL_TYPE"

python -m scripts.run_single_task \
    --dataset EFNY \
    --config ${paths_config} \
    --task_id $SLURM_ARRAY_TASK_ID \
    --model_type $MODEL_TYPE \
    --output_prefix efny_perm_${MODEL_TYPE} \
    --random_state $RANDOM_STATE \
    --log_level INFO \
    --log_file ${project_dir}/outputs/logs/perm_${MODEL_TYPE}/task_${SLURM_ARRAY_TASK_ID}.log

if [ $? -eq 0 ]; then
    echo "Permutation task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Permutation task $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi
