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
ROOT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
SUBLIST="$ROOT_DIR/data/EFNY/table/sublist/rest_valid_sublist.txt"
SCRIPT="$ROOT_DIR/src/functional_conn/compute_fc_schaefer.py"
QC_FILE="$ROOT_DIR/data/EFNY/table/qc/rest_fd_summary.csv"

# Ensure log directory exists (this might fail if running on node, but good to have)
mkdir -p "$ROOT_DIR/outputs/EFNY/logs/functional_conn"

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
        --xcpd-dir "$ROOT_DIR/data/EFNY/MRI_data/xcpd_rest" \
        --out-dir "$ROOT_DIR/data/EFNY/functional_conn/rest" \
        --qc-file "$QC_FILE" \
        --valid-list "$SUBLIST"
done

echo "Finished Subject: $SUBJECT"
