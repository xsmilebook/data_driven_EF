#!/bin/bash
#SBATCH --job-name=efny_real_scca      # jobname
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/real_scca/efny_real_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/log/real_scca/efny_real_%A_%a.err
#SBATCH --partition=q_fat
#SBATCH --cpus-per-task=1
#SBATCH --array=0-10                    # 101 runs of real-data analysis

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
project_dir="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF"
config_file="${project_dir}/src/models/config.json"

MODEL_TYPE="scca"
N_COMPONENTS=3
RANDOM_STATE_BASE=42
ATLAS="schaefer400"
DATASET="EFNT"

TASK_TYPE="real"

mkdir -p ${project_dir}/log/real_scca

echo "Starting REAL run $SLURM_ARRAY_TASK_ID at $(date)"
echo "Model: $MODEL_TYPE"
echo "Number of components: ${N_COMPONENTS}"

python ${project_dir}/src/scripts/run_single_task.py \
    --task_id 0 \
    --model_type $MODEL_TYPE \
    --n_components $N_COMPONENTS \
    --output_prefix efny_real_${MODEL_TYPE}_run_${SLURM_ARRAY_TASK_ID} \
    --random_state $((RANDOM_STATE_BASE + SLURM_ARRAY_TASK_ID)) \
    --config_file ${config_file} \
    --log_level INFO \
    --log_file ${project_dir}/log/real_scca/run_${SLURM_ARRAY_TASK_ID}.log

if [ $? -eq 0 ]; then
    echo "Real run $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
else
    echo "Real run $SLURM_ARRAY_TASK_ID failed at $(date)"
    exit 1
fi

