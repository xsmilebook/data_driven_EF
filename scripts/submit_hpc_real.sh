#!/bin/bash
#SBATCH --job-name=efny_real_adaptive_pls      # jobname
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --output=logs/EFNY/real_adaptive_pls/%x_%A_%a.out
#SBATCH --error=logs/EFNY/real_adaptive_pls/%x_%A_%a.err
#SBATCH --partition=q_fat
#SBATCH --cpus-per-task=1
#SBATCH --array=0-10                    # 11 runs of real-data analysis

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
paths_config="${project_dir}/configs/paths.yaml"

MODEL_TYPE="adaptive_pls"
RANDOM_STATE_BASE=42
DATASET="EFNY"

TASK_TYPE="real"

mkdir -p ${project_dir}/logs/EFNY/real_${MODEL_TYPE}

echo "Starting REAL run $SLURM_ARRAY_TASK_ID at $(date)"
echo "Model: $MODEL_TYPE"

python -m scripts.run_single_task \
    --dataset ${DATASET} \
    --config ${paths_config} \
    --task_id 0 \
    --model_type $MODEL_TYPE \
    --output_prefix efny_real_${MODEL_TYPE}_run_${SLURM_ARRAY_TASK_ID} \
    --random_state $((RANDOM_STATE_BASE + SLURM_ARRAY_TASK_ID)) \
    --log_level INFO \
    --log_file ${project_dir}/logs/EFNY/real_${MODEL_TYPE}/run_${SLURM_ARRAY_TASK_ID}.log

if [ $? -eq 0 ]; then
    echo "Real run $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Real run $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi

