#!/bin/bash
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -p q_fat

module load singularity
subj=$1

eval "$(python -m scripts.render_paths --dataset EFNY --config configs/paths.yaml --format bash)"
fmriprep_Path=${FMRIPREP_DIR}
if [ -z "$fmriprep_Path" ]; then
  echo "ERROR: FMRIPREP_DIR is not set. Configure external_inputs.fmriprep_dir in configs/datasets/EFNY.yaml" 1>&2
  exit 1
fi

xcpd_Path=${INTERIM_ROOT}/MRI_data/xcpd_rest
temp_dir=/ibmgpfs/cuizaixu_lab/xuhaoshu/trash/sub-${subj}
mkdir -p $temp_dir

fslic=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
templateflow=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow
fs_subjects_dir=/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/freesurfer
wd=${INTERIM_ROOT}/wd/xcpd/sub-${subj}
mkdir -p $wd
output=${xcpd_Path}

export SINGULARITYENV_TEMPLATEFLOW_HOME=$templateflow
export SINGULARITYENV_SUBJECTS_DIR=/freesurfer
singularity run --cleanenv \
        -B $fmriprep_Path:/fmriprep \
        -B $output:/output \
        -B $wd:/wd \
        -B $fslic:/fslic \
        -B $fs_subjects_dir:/freesurfer \
        -B $templateflow:$templateflow \
		-B /ibmgpfs/cuizaixu_lab/xuhaoshu/tmp:/tmp \
        -B $temp_dir:$HOME \
        /ibmgpfs/cuizaixu_lab/congjing/singularity/xcp_d-0.10.0.simg \
        /fmriprep /output participant \
        --input-type fmriprep \
        --mode none \
        --participant_label ${subj} --task-id rest \
        --fs-license-file /fslic/license.txt \
        -w /wd --nthreads 2 --mem-gb 40 \
        --nuisance-regressors 36P \
        --despike \
        --lower-bpf=0.01 --upper-bpf=0.1 \
        --motion-filter-type lp --band-stop-min 6 \
        --fd-thresh -1

rm -rf $wd
