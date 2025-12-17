import os

import numpy as np
import pandas as pd

from .io import normalize_columns, read_task_config, subject_code_from_filename
from .registry import KNOWN_METRICS, METRIC_FUNCS, REQUIRED_COLUMNS, SKIP_TASKS


def process_file(path, task_rows):
    res = {'subject_code': subject_code_from_filename(path), 'file_name': os.path.basename(path)}
    try:
        xl = pd.ExcelFile(path)
        sheets = set(xl.sheet_names)
    except Exception:
        return res

    for item in task_rows:
        task = item['Task']
        if task in SKIP_TASKS:
            continue
        metrics = item['metrics']
        if len(metrics) == 0:
            continue

        if task not in sheets:
            for m in metrics:
                res[f'{task}_{m}'] = np.nan
            continue

        try:
            df = xl.parse(task, dtype='object')
        except Exception:
            for m in metrics:
                res[f'{task}_{m}'] = np.nan
            continue
        df = normalize_columns(df)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        for m in metrics:
            func = METRIC_FUNCS.get(m)
            val = np.nan if len(missing) > 0 and m in KNOWN_METRICS else (func(df) if func is not None else np.nan)
            res[f'{task}_{m}'] = val
    return res


def run(data_dir, task_csv, out_csv):
    tasks = read_task_config(task_csv)
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith('.xlsx'):
                files.append(os.path.join(root, fn))
    files.sort()
    rows = []
    for fp in files:
        rows.append(process_file(fp, tasks))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding='utf-8')
