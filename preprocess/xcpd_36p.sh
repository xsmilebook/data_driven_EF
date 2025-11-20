#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -p q_cn

module load singularity
subj=$1

fmriprep_Path=/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/results/fmriprep_rest

xcpd_Path=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/MRI_data/xcpd_rest
temp_dir=/ibmgpfs/cuizaixu_lab/xuhaoshu/trash/${label}
mkdir -p $temp_dir

fslic=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
templateflow=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow
wd=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/wd/xcpd/${subj}
output=${xcpd_Path}

export SINGULARITYENV_TEMPLATEFLOW_HOME=$templateflow
singularity run --cleanenv \
        -B $fmriprep_Path:/fmriprep \
        -B $output:/output \
        -B $wd:/wd \
        -B $fslic:/fslic \
        -B $templateflow:$templateflow \
		-B /ibmgpfs/cuizaixu_lab/xuhaoshu/tmp:/tmp \
        -B $temp_dir:$HOME \
        /ibmgpfs/cuizaixu_lab/xulongzhou/apps/singularity/xcpd-0.7.1rc5.simg \
        /fmriprep /output participant \
        --participant_label ${subj} --task-id rest \
        --fs-license-file /fslic/license.txt \
        -w /wd --nthreads 2 --mem-gb 40 \
        --nuisance-regressors 36P \
        --despike \
        --lower-bpf=0.01 --upper-bpf=0.1 \
        --motion-filter-type lp --band-stop-min 6 \
        --fd-thresh -1

rm -rf $wd