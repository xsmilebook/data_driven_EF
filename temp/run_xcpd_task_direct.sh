#!/usr/bin/env bash
#SBATCH --chdir=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH -p q_fat_c

set -euo pipefail

PROJECT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"

RAW_ROOT="${PROJECT_DIR}/data/raw"
INTERIM_ROOT="${PROJECT_DIR}/data/interim"

TASK_PSYCH_DIR="${RAW_ROOT}/MRI_data/task_psych"
FMRIPREP_TASK_NBACK_DIR="/ibmgpfs/cuizaixu_lab/xuxiaoyu/songxinwei/fmri_task/Tsinghua/Tsinghua_nback/fmriprep_task"
FMRIPREP_TASK_SST_DIR="/ibmgpfs/cuizaixu_lab/xuxiaoyu/songxinwei/fmri_task/Tsinghua/Tsinghua_sst/fmriprep_task"
FMRIPREP_TASK_SWITCH_DIR="/ibmgpfs/cuizaixu_lab/xuxiaoyu/songxinwei/fmri_task/Tsinghua/Tsinghua_switch/fmriprep_task"

XCPD_IMAGE="/ibmgpfs/cuizaixu_lab/xulongzhou/apps/singularity/xcpd-0.7.1rc5.simg"
FREESURFER_DIR="/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer"
TEMPLATEFLOW_DIR="/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow"
TMP_BIND="/ibmgpfs/cuizaixu_lab/xuhaoshu/tmp"
HOME_TRASH_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/trash"

subj="${1:-}"
task="${2:-}"
if [ -z "${subj}" ] || [ -z "${task}" ]; then
  echo "Usage: $0 <SUBJECT_LABEL> <TASK: nback|sst|switch>" 1>&2
  exit 2
fi

if [ -d "${PROJECT_DIR}" ]; then
  cd "${PROJECT_DIR}"
else
  echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}" 1>&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load singularity || true
fi

subj_label="${subj#sub-}"
task="$(echo "${task}" | tr '[:upper:]' '[:lower:]')"
case "${task}" in
  nback|sst|switch) ;;
  *) echo "Invalid task: ${task} (expected nback|sst|switch)" 1>&2; exit 2 ;;
esac

case "${task}" in
  nback) fmriprep_Path="${FMRIPREP_TASK_NBACK_DIR}" ;;
  sst) fmriprep_Path="${FMRIPREP_TASK_SST_DIR}" ;;
  switch) fmriprep_Path="${FMRIPREP_TASK_SWITCH_DIR}" ;;
esac

if [ ! -d "${fmriprep_Path}" ]; then
  echo "ERROR: fMRIPrep dir not found: ${fmriprep_Path}" 1>&2
  exit 1
fi
if [ ! -d "${TASK_PSYCH_DIR}" ]; then
  echo "ERROR: TASK_PSYCH_DIR not found: ${TASK_PSYCH_DIR}" 1>&2
  exit 1
fi
if [ ! -f "${XCPD_IMAGE}" ]; then
  echo "ERROR: XCP-D image not found: ${XCPD_IMAGE}" 1>&2
  exit 1
fi
if [ ! -f "${FREESURFER_DIR}/license.txt" ]; then
  echo "ERROR: FreeSurfer license not found: ${FREESURFER_DIR}/license.txt" 1>&2
  exit 1
fi

output="${INTERIM_ROOT}/MRI_data/xcpd_task/${task}"
custom_confounds_root="${INTERIM_ROOT}/MRI_data/xcpd_task/custom_confounds/${task}/sub-${subj_label}"
wd="${INTERIM_ROOT}/wd/xcpd_task/${task}/sub-${subj_label}"
mkdir -p "${output}" "${custom_confounds_root}" "${wd}"

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

temp_dir="${HOME_TRASH_ROOT}/sub-${subj_label}"
mkdir -p "${temp_dir}"

export SINGULARITYENV_TEMPLATEFLOW_HOME="${TEMPLATEFLOW_DIR}"

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
  fmriprep_bind_src="${wrapper}"
  if [ -d "${fmriprep_input}/sub-${subj_label}" ]; then
    fmriprep_bind_extra="-B ${fmriprep_input}/sub-${subj_label}:/fmriprep/sub-${subj_label}"
  else
    fmriprep_bind_extra="-B ${fmriprep_input}:/fmriprep/sub-${subj_label}"
  fi
fi

singularity run --cleanenv \
  -B "${fmriprep_bind_src}:/fmriprep" \
  -B "${output}:/output" \
  -B "${wd}:/wd" \
  -B "${FREESURFER_DIR}:/fslic" \
  -B "${TEMPLATEFLOW_DIR}:${TEMPLATEFLOW_DIR}" \
  -B "${custom_confounds_root}:/custom_confounds" \
  -B "${TMP_BIND}:/tmp" \
  -B "${temp_dir}:${HOME}" \
  ${fmriprep_bind_extra} \
  "${XCPD_IMAGE}" \
  /fmriprep /output participant \
  --participant_label "${subj_label}" --task-id "${task}" \
  --nuisance-regressors 36P \
  --custom_confounds /custom_confounds \
  --fs-license-file /fslic/license.txt \
  -w /wd --nthreads 3 --mem-gb 60 \
  --despike \
  --lower-bpf=0.01 --upper-bpf=0.1 \
  --motion-filter-type lp --band-stop-min 6 \
  --fd-thresh -1

rm -rf "${wd}"
