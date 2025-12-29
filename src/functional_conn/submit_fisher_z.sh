#!/bin/bash
#SBATCH --job-name=fisher_z_fc
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=q_fat
#SBATCH --array=1-508%508
#SBATCH --output=outputs/EFNY/logs/functional_conn_z/%x_%A_%a.out
#SBATCH --error=outputs/EFNY/logs/functional_conn_z/%x_%A_%a.err

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

ROOT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
SUBLIST="$ROOT_DIR/data/EFNY/table/sublist/rest_valid_sublist.txt"
SCRIPT="$ROOT_DIR/src/functional_conn/fisher_z_fc.py"

mkdir -p "$ROOT_DIR/outputs/EFNY/logs/functional_conn_z"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBLIST" | tr -d '\r')
if [ -z "$SUBJECT" ]; then
    echo "Error: No subject found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

for ROIS in 100 200 400; do
    python "$SCRIPT" \
        --subject "$SUBJECT" \
        --n-rois $ROIS \
        --in-dir "$ROOT_DIR/data/EFNY/functional_conn/rest" \
        --out-dir "$ROOT_DIR/data/EFNY/functional_conn_z/rest"
done

