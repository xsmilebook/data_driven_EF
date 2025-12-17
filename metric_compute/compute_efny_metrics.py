import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT, 'data', 'EFNY', 'behavior_data', 'cibr_app_data')
TABLE_DIR = os.path.join(ROOT, 'data', 'EFNY', 'table', 'metrics')
TASK_CSV = os.path.join(TABLE_DIR, 'EFNY_task.csv')
OUT_CSV = os.path.join(TABLE_DIR, 'EFNY_metrics.csv')

from efny.io import first_col, normalize_columns, read_task_config, subject_code_from_filename
from efny.main import process_file
from efny.metrics import (
    metric_acc,
    metric_contrast_acc,
    metric_contrast_rt,
    metric_dprime,
    metric_rt,
    metric_ssrt,
    metric_switch_cost,
)
from efny.preprocess import RT_MAX, RT_MIN, clean_rt, determine_correct, to_numeric
from efny.registry import KNOWN_METRICS, METRIC_FUNCS, REQUIRED_COLUMNS, SKIP_TASKS

def main():
    from efny.main import run

    run(data_dir=DATA_DIR, task_csv=TASK_CSV, out_csv=OUT_CSV)

if __name__ == '__main__':
    main()
