#!/bin/bash

eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
SUBLIST_FILE="${PROCESSED_ROOT}/table/sublist/mri_sublist.txt"
LOG_DIR="${LOGS_ROOT}/${DATASET}/xcpd"
mkdir -p "${LOG_DIR}"

for subj in `cat ${SUBLIST_FILE}`
do  
    #subj=$(echo $file | awk -F'_' '{print $1}')
    echo "perform xcpd of subject: $subj"
    sbatch --chdir="${PROJECT_DIR}" -J ${subj} -o ${LOG_DIR}/out.${subj}.txt -e ${LOG_DIR}/error.${subj}.txt src/preprocess/xcpd_36p.sh $subj
done
