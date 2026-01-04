#!/bin/bash

eval "$(python -m scripts.render_paths --dataset THU_TASK --config configs/paths.yaml --dataset-config configs/dataset_tsinghua_taskfmri.yaml --format bash)"

SUBLIST_FILE=${1:-"${PROCESSED_ROOT}/table/sublist/taskfmri_sublist.txt"}
TASKS=${2:-"nback sst switch"}

LOG_DIR="${LOGS_ROOT}/${DATASET}/xcpd_task"
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
      src/imaging_preprocess/xcpd_task_36p_taskreg.sh $subj $task
  done
done

