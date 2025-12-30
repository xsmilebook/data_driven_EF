#!/bin/bash
#SBATCH --job-name=convert_fc_vectors_efny
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=q_fat_c
#SBATCH --output=outputs/EFNY/logs/fc_vector/%x_%A_%a.out
#SBATCH --error=outputs/EFNY/logs/fc_vector/%x_%A_%a.err
# NOTE: SBATCH log paths are static (no env expansion). Keep dataset-specific paths here.

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

# Set up paths for EFNY dataset
eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
PROJECT_ROOT="${PROJECT_DIR}"
INPUT_PATH="${INTERIM_ROOT}/functional_conn_z/rest"
SUBLIST_FILE="${PROCESSED_ROOT}/table/sublist/sublist.txt"
OUTPUT_PATH="${PROCESSED_ROOT}/fc_vector"
DATASET_NAME="${DATASET}"
LOG_DIR="${LOGS_ROOT}/${DATASET}/fc_vector"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${LOG_DIR}"

# Run the conversion script for different ROI numbers
echo "Starting FC matrix to vector conversion for ${DATASET_NAME}"
echo "Input path: ${INPUT_PATH}"
echo "Subject list: ${SUBLIST_FILE}"
echo "Output path: ${OUTPUT_PATH}"

for N_ROIS in 100 200 400; do
    echo "Processing Schaefer${N_ROIS} ROIs..."
    python ${PROJECT_ROOT}/src/functional_conn/convert_fc_vector.py \
        --input_path "${INPUT_PATH}" \
        --sublist_file "${SUBLIST_FILE}" \
        --output_path "${OUTPUT_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --n_rois ${N_ROIS}
done

echo "FC matrix to vector conversion completed"
