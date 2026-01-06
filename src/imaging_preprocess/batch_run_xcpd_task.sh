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

for subj in `cat ${SUBLIST_FILE}`
do
  subj=$(echo "$subj" | tr -d '\r')
  if [ -z "$subj" ]; then
    continue
  fi
  for task in ${TASKS}
  do
    echo "perform xcpd task regression: subject=${subj} task=${task}"
    sbatch --chdir="${PROJECT_DIR}" \
      -J ${subj}_${task} \
      -o ${LOG_DIR}/out.${task}.${subj}.txt \
      -e ${LOG_DIR}/error.${task}.${subj}.txt \
      src/imaging_preprocess/xcpd_task_36p_taskreg.sh $subj $task ${DATASET_CONFIG} ${DATASET_NAME}
  done
done
