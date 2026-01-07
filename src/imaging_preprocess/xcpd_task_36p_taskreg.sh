#!/bin/bash
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p q_fat_c

# network settings for compute note
export http_proxy=http://10.11.100.5:3128
export HTTP_PROXY=http://10.11.100.5:3128
export https_proxy=http://10.11.100.5:3128
export HTTPS_PROXY=http://10.11.100.5:3128
export ftp_proxy=http://10.11.100.5:3128
export FTP_PROXY=http://10.11.100.5:3128
export all_proxy=http://10.11.100.5:3128
export ALL_PROXY=http://10.11.100.5:3128

module load singularity

subj=$1
task=$2
dataset_config=${3:-configs/dataset_tsinghua_taskfmri.yaml}
dataset_name=${4:-EFNY_THU}
if [ -z "$subj" ] || [ -z "$task" ]; then
  echo "Usage: $0 <SUBJECT_LABEL> <TASK: nback|sst|switch> [DATASET_CONFIG] [DATASET_NAME]" 1>&2
  exit 2
fi

subj_label=${subj#sub-}

task=$(echo "$task" | tr '[:upper:]' '[:lower:]')
case "$task" in
  nback|sst|switch) ;;
  *) echo "Invalid task: $task (expected nback|sst|switch)" 1>&2; exit 2 ;;
esac

eval "$(python -m scripts.render_paths --dataset ${dataset_name} --config configs/paths.yaml --dataset-config ${dataset_config} --format bash)"

case "$task" in
  nback) fmriprep_Path=${FMRIPREP_TASK_NBACK_DIR} ;;
  sst) fmriprep_Path=${FMRIPREP_TASK_SST_DIR} ;;
  switch) fmriprep_Path=${FMRIPREP_TASK_SWITCH_DIR} ;;
esac

if [ -z "$fmriprep_Path" ]; then
  echo "ERROR: fmriprep path for task=$task is not configured in ${dataset_config}" 1>&2
  exit 1
fi
if [ -z "$TASK_PSYCH_DIR" ]; then
  echo "ERROR: TASK_PSYCH_DIR is not configured in ${dataset_config}" 1>&2
  exit 1
fi

xcpd_task_dirname=${XCPD_TASK_DIRNAME:-xcpd_task}
output=${INTERIM_ROOT}/MRI_data/${xcpd_task_dirname}/${task}
custom_confounds_root=${INTERIM_ROOT}/MRI_data/${xcpd_task_dirname}/custom_confounds/${task}/sub-${subj_label}
wd=${INTERIM_ROOT}/wd/${xcpd_task_dirname}/${task}/sub-${subj_label}
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

if [ ! -f "${custom_confounds_root}/confounds_config.yml" ]; then
  echo "ERROR: custom confounds config not found: ${custom_confounds_root}/confounds_config.yml" 1>&2
  echo "Check whether task_psych_dir and subject naming match, and whether the behavior CSV exists for task=${task}." 1>&2
  exit 1
fi

temp_dir=/ibmgpfs/cuizaixu_lab/xuhaoshu/trash/sub-${subj_label}
mkdir -p "$temp_dir"

fslic=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
fs_subjects_dir=${FREESURFER_SUBJECTS_DIR:-/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/freesurfer}
templateflow=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow
export SINGULARITYENV_TEMPLATEFLOW_HOME=$templateflow
export SINGULARITYENV_SUBJECTS_DIR=/freesurfer

# Pass proxy into the container (this script uses `singularity run --cleanenv`).
# Note: urllib3 requires proxy URLs to include scheme (http:// or https://).
proxy_url="http://10.11.100.5:3128"
export SINGULARITYENV_http_proxy="${proxy_url}"
export SINGULARITYENV_HTTP_PROXY="${proxy_url}"
export SINGULARITYENV_https_proxy="${proxy_url}"
export SINGULARITYENV_HTTPS_PROXY="${proxy_url}"
export SINGULARITYENV_ftp_proxy="${proxy_url}"
export SINGULARITYENV_FTP_PROXY="${proxy_url}"
export SINGULARITYENV_all_proxy="${proxy_url}"
export SINGULARITYENV_ALL_PROXY="${proxy_url}"

fmriprep_bind_src="${fmriprep_input}"
fmriprep_bind_extra=""
if [ ! -f "${fmriprep_input}/dataset_description.json" ]; then
  wrapper="${wd}/fmriprep_wrapper"
  mkdir -p "${wrapper}"
  cat > "${wrapper}/dataset_description.json" <<EOF
{
  "Name": "fMRIPrep derivatives wrapper",
  "BIDSVersion": "1.6.0",
  "DatasetType": "derivative",
  "GeneratedBy": [{"Name": "data_driven_EF/xcpd_task_36p_taskreg.sh"}]
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
  -w /wd --nthreads 4 --mem-gb 80 \
  --file-format cifti \
  --output-type censored \
  --combine-runs n \
  --warp-surfaces-native2std n \
  --linc-qc n \
  --abcc-qc n \
  --min-coverage 0 \
  --create-matrices all \
  --head-radius 50 \
  --bpf-order 2 \
  --resource-monitor \
  --smoothing 2 \
  --despike n \
  --lower-bpf 0.01 --upper-bpf 0.1 \
  --motion-filter-type lp --band-stop-min 6 \
  --fd-thresh 100

rm -rf "$wd"
