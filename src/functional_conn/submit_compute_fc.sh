#!/bin/bash
#SBATCH --job-name=compute_fc
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=q_cn
#SBATCH --array=1-508%508            # TODO: Update array range based on line count of sublist
#SBATCH --output=outputs/EFNY/logs/functional_conn/%x_%A_%a.out
#SBATCH --error=outputs/EFNY/logs/functional_conn/%x_%A_%a.err
# NOTE: SBATCH log paths are static (no env expansion). Keep dataset-specific paths here.

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

export http_proxy=10.11.100.5:3128 
export HTTP_PROXY=10.11.100.5:3128
export https_proxy=10.11.100.5:3128
export HTTPS_PROXY=10.11.100.5:3128
export ftp_proxy=10.11.100.5:3128
export FTP_PROXY=10.11.100.5:3128
export all_proxy=10.11.100.5:3128
export ALL_PROXY=10.11.100.5:3128

# Define paths
eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
ROOT_DIR="${PROJECT_DIR}"
SUBLIST="$PROCESSED_ROOT/table/sublist/rest_valid_sublist.txt"
SCRIPT="$ROOT_DIR/src/functional_conn/compute_fc_schaefer.py"
QC_FILE="$INTERIM_ROOT/table/qc/rest_fd_summary.csv"
LOG_DIR="$OUTPUTS_ROOT/logs/functional_conn"

# Ensure log directory exists (this might fail if running on node, but good to have)
mkdir -p "$LOG_DIR"

# Get subject from the list based on the array task ID
# sed -n 'Np' prints the Nth line
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBLIST" | tr -d '\r')

if [ -z "$SUBJECT" ]; then
    echo "Error: No subject found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing Subject: $SUBJECT"
echo "Date: $(date)"

# Loop through desired resolutions
for ROIS in 100 200 400; do
    echo "Running Schaefer${ROIS}..."
    python "$SCRIPT" \
        --subject "$SUBJECT" \
        --n-rois $ROIS \
        --xcpd-dir "$INTERIM_ROOT/MRI_data/xcpd_rest" \
        --out-dir "$INTERIM_ROOT/functional_conn/rest" \
        --qc-file "$QC_FILE" \
        --valid-list "$SUBLIST"
done

echo "Finished Subject: $SUBJECT"
