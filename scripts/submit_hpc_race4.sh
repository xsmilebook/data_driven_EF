#!/bin/bash
#SBATCH --job-name=efny_race4
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --output=outputs/logs/race4/%x_%A_%a.out
#SBATCH --error=outputs/logs/race4/%x_%A_%a.err
#SBATCH --partition=q_fat
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
paths_config="${project_dir}/configs/paths.yaml"

DATASET="EFNY"

mkdir -p ${project_dir}/outputs/logs/race4

echo "Starting RACE4 job $SLURM_ARRAY_TASK_ID at $(date)"

python -m scripts.fit_race4_task \
    --dataset ${DATASET} \
    --config ${paths_config} \
    --job-index ${SLURM_ARRAY_TASK_ID} \
    --draws 500 \
    --tune 500 \
    --chains 2 \
    --target-accept 0.95 \
    --seed 1

if [ $? -eq 0 ]; then
    echo "RACE4 job $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "RACE4 job $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi

