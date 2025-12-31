#!/bin/bash
#SBATCH --job-name=fisher_z_fc
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=q_fat
#SBATCH --array=1-508%508
#SBATCH --output=outputs/logs/functional_conn_z/%x_%A_%a.out
#SBATCH --error=outputs/logs/functional_conn_z/%x_%A_%a.err
# NOTE: SBATCH log paths are static (no env expansion). Keep dataset-specific paths here.

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
ROOT_DIR="${PROJECT_DIR}"
SUBLIST="$PROCESSED_ROOT/table/sublist/rest_valid_sublist.txt"
SCRIPT="$ROOT_DIR/src/functional_conn/fisher_z_fc.py"
LOG_DIR="$LOGS_ROOT/${DATASET}/functional_conn_z"

mkdir -p "$LOG_DIR"

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBLIST" | tr -d '\r')
if [ -z "$SUBJECT" ]; then
    echo "Error: No subject found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

for ROIS in 100 200 400; do
    python "$SCRIPT" \
        --subject "$SUBJECT" \
        --n-rois $ROIS \
        --in-dir "$INTERIM_ROOT/functional_conn/rest" \
        --out-dir "$INTERIM_ROOT/functional_conn_z/rest"
done

