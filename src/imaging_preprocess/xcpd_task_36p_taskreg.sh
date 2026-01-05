#!/bin/bash
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH -p q_fat_c

module load singularity

subj=$1
task=$2
if [ -z "$subj" ] || [ -z "$task" ]; then
  echo "Usage: $0 <SUBJECT_LABEL> <TASK: nback|sst|switch>" 1>&2
  exit 2
fi

subj_label=${subj#sub-}

task=$(echo "$task" | tr '[:upper:]' '[:lower:]')
case "$task" in
  nback|sst|switch) ;;
  *) echo "Invalid task: $task (expected nback|sst|switch)" 1>&2; exit 2 ;;
esac

eval "$(python -m scripts.render_paths --dataset THU_TASK --config configs/paths.yaml --dataset-config configs/dataset_tsinghua_taskfmri.yaml --format bash)"

case "$task" in
  nback) fmriprep_Path=${FMRIPREP_TASK_NBACK_DIR} ;;
  sst) fmriprep_Path=${FMRIPREP_TASK_SST_DIR} ;;
  switch) fmriprep_Path=${FMRIPREP_TASK_SWITCH_DIR} ;;
esac

if [ -z "$fmriprep_Path" ]; then
  echo "ERROR: fmriprep path for task=$task is not configured in configs/dataset_tsinghua_taskfmri.yaml" 1>&2
  exit 1
fi
if [ -z "$TASK_PSYCH_DIR" ]; then
  echo "ERROR: TASK_PSYCH_DIR is not configured in configs/dataset_tsinghua_taskfmri.yaml" 1>&2
  exit 1
fi

output=${INTERIM_ROOT}/MRI_data/xcpd_task/${task}
custom_confounds_root=${INTERIM_ROOT}/MRI_data/xcpd_task/custom_confounds/${task}/sub-${subj_label}
wd=${INTERIM_ROOT}/wd/xcpd_task/${task}/sub-${subj_label}
mkdir -p "$output" "$custom_confounds_root" "$wd"

fmriprep_input=""
if [ -d "${fmriprep_Path}/sub-${subj_label}/fmriprep" ]; then
  fmriprep_input="${fmriprep_Path}/sub-${subj_label}/fmriprep"
elif [ -d "${fmriprep_Path}/sub-${subj_label}" ]; then
  fmriprep_input="${fmriprep_Path}"
elif [ -d "${fmriprep_Path}/fmriprep" ]; then
  fmriprep_input="${fmriprep_Path}/fmriprep"
else
  fmriprep_input="${fmriprep_Path}"
fi

python -m scripts.build_task_xcpd_confounds \
  --subject "${subj}" \
  --task "${task}" \
  --out-root "${custom_confounds_root}" \
  --fmriprep-dir "${fmriprep_input}" \
  --task-psych-dir "${TASK_PSYCH_DIR}" \
  --fir-window-seconds 20

temp_dir=/ibmgpfs/cuizaixu_lab/xuhaoshu/trash/sub-${subj_label}
mkdir -p "$temp_dir"

fslic=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
fs_subjects_dir=/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/freesurfer
templateflow=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow
export SINGULARITYENV_TEMPLATEFLOW_HOME=$templateflow
export SINGULARITYENV_SUBJECTS_DIR=/freesurfer

fmriprep_bind_src="${fmriprep_input}"
fmriprep_bind_extra=""
if [ ! -f "${fmriprep_input}/dataset_description.json" ]; then
  wrapper="${wd}/fmriprep_wrapper"
  mkdir -p "${wrapper}"
  cat > "${wrapper}/dataset_description.json" <<EOF
{
  "Name": "fMRIPrep derivatives wrapper",
  "BIDSVersion": "1.6.0",
  "DatasetType": "derivative"
}
EOF
  # Bind the subject folder into a wrapper root that includes dataset_description.json,
  # without writing into the original fMRIPrep directory.
  fmriprep_bind_src="${wrapper}"
  if [ -d "${fmriprep_input}/sub-${subj_label}" ]; then
    fmriprep_bind_extra="-B ${fmriprep_input}/sub-${subj_label}:/fmriprep/sub-${subj_label}"
  else
    fmriprep_bind_extra="-B ${fmriprep_input}:/fmriprep/sub-${subj_label}"
  fi
fi

singularity run --cleanenv \
  -B $fmriprep_bind_src:/fmriprep \
  -B $output:/output \
  -B $wd:/wd \
  -B $fslic:/fslic \
  -B $fs_subjects_dir:/freesurfer \
  -B $templateflow:$templateflow \
  -B $custom_confounds_root:/custom_confounds \
  -B /ibmgpfs/cuizaixu_lab/xuhaoshu/tmp:/tmp \
  -B $temp_dir:$HOME \
  ${fmriprep_bind_extra} \
  /ibmgpfs/cuizaixu_lab/congjing/singularity/xcp_d-0.10.0.simg \
  /fmriprep /output participant \
  --input-type fmriprep \
  --mode none \
  --participant_label ${subj_label} --task-id ${task} \
  --datasets custom=/custom_confounds \
  --nuisance-regressors /custom_confounds/confounds_config.yml \
  --fs-license-file /fslic/license.txt \
  -w /wd --nthreads 3 --mem-gb 60 \
  --despike \
  --lower-bpf=0.01 --upper-bpf=0.1 \
  --motion-filter-type lp --band-stop-min 6 \
  --fd-thresh -1

rm -rf "$wd"
