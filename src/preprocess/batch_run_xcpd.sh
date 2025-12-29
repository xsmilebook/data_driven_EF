#!/bin/bash

PROJECT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
SUBLIST_FILE="${PROJECT_DIR}/data/EFNY/table/sublist/mri_sublist.txt"
LOG_DIR="${PROJECT_DIR}/outputs/EFNY/logs/xcpd"
mkdir -p "${LOG_DIR}"

for subj in `cat ${SUBLIST_FILE}`
do  
    #subj=$(echo $file | awk -F'_' '{print $1}')
    echo "perform xcpd of subject: $subj"
    sbatch -J ${subj} -o ${LOG_DIR}/out.${subj}.txt -e ${LOG_DIR}/error.${subj}.txt xcpd_36p.sh $subj
done
