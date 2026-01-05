#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/data_driven_EF"
PROCESSED_ROOT="${PROJECT_DIR}/data/processed"
LOGS_ROOT="${PROJECT_DIR}/outputs/logs"
DATASET="THU_TASK"

SUBLIST_FILE="${1:-${PROCESSED_ROOT}/table/sublist/taskfmri_sublist.txt}"
TASKS="${2:-nback sst switch}"

LOG_DIR="${LOGS_ROOT}/${DATASET}/xcpd_task"
mkdir -p "${LOG_DIR}"

if [ ! -f "${SUBLIST_FILE}" ]; then
  echo "ERROR: subject list not found: ${SUBLIST_FILE}" 1>&2
  exit 1
fi

if [ ! -f "${PROJECT_DIR}/temp/run_xcpd_task_direct.sh" ]; then
  echo "ERROR: missing direct single-job script: ${PROJECT_DIR}/temp/run_xcpd_task_direct.sh" 1>&2
  exit 1
fi

cd "${PROJECT_DIR}"

while IFS= read -r subj; do
  subj="$(echo "${subj}" | tr -d '\r' | xargs)"
  if [ -z "${subj}" ]; then
    continue
  fi
  for task in ${TASKS}; do
    echo "perform xcpd task regression: subject=${subj} task=${task}"
    sbatch --chdir="${PROJECT_DIR}" \
      -J "${subj}_${task}" \
      -o "${LOG_DIR}/out.${task}.${subj}.txt" \
      -e "${LOG_DIR}/error.${task}.${subj}.txt" \
      temp/run_xcpd_task_direct.sh "${subj}" "${task}"
  done
done < "${SUBLIST_FILE}"

