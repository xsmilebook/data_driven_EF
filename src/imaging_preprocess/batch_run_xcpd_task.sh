#!/bin/bash

SUBLIST_FILE=${1:-""}
TASKS=${2:-"nback sst switch"}
DATASET_CONFIG=${3:-configs/dataset_tsinghua_taskfmri.yaml}
DATASET_NAME=${4:-EFNY_THU}

eval "$(python -m scripts.render_paths --dataset ${DATASET_NAME} --config configs/paths.yaml --dataset-config ${DATASET_CONFIG} --format bash)"

if [ -z "${SUBLIST_FILE}" ]; then
  SUBLIST_FILE="${PROCESSED_ROOT}/table/sublist/taskfmri_sublist.txt"
fi

xcpd_task_dirname=${XCPD_TASK_DIRNAME:-xcpd_task}
LOG_DIR="${LOGS_ROOT}/${DATASET}/${xcpd_task_dirname}"
mkdir -p "${LOG_DIR}"

if [ ! -f "${SUBLIST_FILE}" ]; then
  echo "ERROR: subject list not found: ${SUBLIST_FILE}" 1>&2
  exit 1
fi

is_xcpd_task_done () {
  local subj_label="$1"
  local task="$2"
  local task_root="${INTERIM_ROOT}/MRI_data/${xcpd_task_dirname}/${task}"
  local subdir="${task_root}/sub-${subj_label}"
  local html="${task_root}/sub-${subj_label}.html"

  if [ ! -d "${subdir}" ] || [ ! -f "${html}" ]; then
    return 1
  fi

  # Require a denoised timeseries and the xcp-d config log for "success".
  shopt -s nullglob
  local denoised=( "${subdir}/func/sub-${subj_label}_task-${task}"_*"desc-denoised_bold.dtseries.nii" )
  local toml=( "${subdir}/log/"*/xcp_d.toml )
  shopt -u nullglob

  if [ ${#denoised[@]} -ge 1 ] && [ ${#toml[@]} -ge 1 ]; then
    return 0
  fi
  return 1
}

for subj in `cat ${SUBLIST_FILE}`
do
  subj=$(echo "$subj" | tr -d '\r')
  if [ -z "$subj" ]; then
    continue
  fi
  subj_label=${subj#sub-}
  for task in ${TASKS}
  do
    if [ -z "${XCPD_FORCE}" ] && is_xcpd_task_done "${subj_label}" "${task}"; then
      echo "skip: already processed subject=${subj_label} task=${task} under ${xcpd_task_dirname}"
      continue
    fi
    echo "perform xcpd task regression: subject=${subj} task=${task}"
    sbatch --chdir="${PROJECT_DIR}" \
      -J ${subj}_${task} \
      -o ${LOG_DIR}/out.${task}.${subj}.txt \
      -e ${LOG_DIR}/error.${task}.${subj}.txt \
      src/imaging_preprocess/xcpd_task_36p_taskreg.sh $subj $task ${DATASET_CONFIG} ${DATASET_NAME}
  done
done
