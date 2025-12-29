#!/bin/bash
#SBATCH --job-name=convert_fc_vectors_efny
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=q_fat_c
#SBATCH --output=outputs/EFNY/logs/fc_vector/%x_%A_%a.out
#SBATCH --error=outputs/EFNY/logs/fc_vector/%x_%A_%a.err

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

# Set up paths for EFNY dataset
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
INPUT_PATH="${PROJECT_ROOT}/data/EFNY/functional_conn_z/rest"
SUBLIST_FILE="${PROJECT_ROOT}/data/EFNY/table/sublist/sublist.txt"
OUTPUT_PATH="${PROJECT_ROOT}/data/EFNY/fc_vector"
DATASET_NAME="EFNY"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"

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
