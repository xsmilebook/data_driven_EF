#!/bin/bash

for subj in `cat /ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/table/sublist/mri_sublist.txt`
do  
    #subj=$(echo $file | awk -F'_' '{print $1}')
    echo "perform xcpd of subject: $subj"
    sbatch -J ${subj} -o /ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/log/xcpd/out.${subj}.txt -e /ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/log/xcpd/error.${subj}.txt xcpd_36p.sh $subj
done
