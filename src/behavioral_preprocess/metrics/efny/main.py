import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import normalize_columns, subject_code_from_filename
from .metrics import get_raw_metrics
from src.config_io import load_simple_yaml


def normalize_task_name(sheet_name: str) -> str:
    s = str(sheet_name)
    low = s.lower()
    if '1back' in low:
        if 'number' in low:
            return 'oneback_number'
        if 'spatial' in low:
            return 'oneback_spatial'
        if 'emotion' in low:
            return 'oneback_emotion'
        return 'oneback_raw'
    if '2back' in low:
        if 'number' in low:
            return 'twoback_number'
        if 'spatial' in low:
            return 'twoback_spatial'
        if 'emotion' in low:
            return 'twoback_emotion'
        return 'twoback_raw'
    return s


DEFAULT_METRICS_CONFIG_PATH = "configs/behavioral_metrics.yaml"


def load_metrics_config(metrics_config_path: str | None = None) -> dict:
    config_path = Path(metrics_config_path or DEFAULT_METRICS_CONFIG_PATH)
    cfg = load_simple_yaml(config_path)
    metrics_cfg = cfg.get("behavioral_metrics", {})
    if not isinstance(metrics_cfg, dict):
        raise ValueError("Missing behavioral_metrics section in metrics config.")
    return metrics_cfg


def load_task_config(metrics_config_path: str | None = None) -> dict:
    metrics_cfg = load_metrics_config(metrics_config_path)
    tasks = metrics_cfg.get("compute_tasks", {})
    if not tasks:
        raise ValueError("Missing behavioral_metrics.compute_tasks in metrics config.")
    return tasks


def load_use_metrics(metrics_config_path: str | None = None) -> dict:
    metrics_cfg = load_metrics_config(metrics_config_path)
    return metrics_cfg.get("use_metrics", {})


def process_file_raw(path, task_config=None):
    if task_config is None:
        task_config = load_task_config()
    res = {'subject_code': subject_code_from_filename(path), 'file_name': os.path.basename(path)}
    try:
        xl = pd.ExcelFile(path)
        sheets = list(xl.sheet_names)
    except Exception:
        return res

    for sh in sheets:
        task_name = normalize_task_name(sh)
        cfg = task_config.get(task_name)
        if cfg is None:
            continue
        try:
            df = xl.parse(sh, dtype='object')
        except Exception:
            for m in cfg.get('metrics', []):
                res[f'{task_name}_{m}'] = np.nan
            continue
        df = normalize_columns(df)

        vals = get_raw_metrics(df, cfg, task_name=task_name)
        for k, v in vals.items():
            res[f'{task_name}_{k}'] = v
    return res


def run_raw(data_dir, out_csv, task_config=None, metrics_config_path=None):
    if task_config is None:
        task_config = load_task_config(metrics_config_path)
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith('.xlsx'):
                files.append(os.path.join(root, fn))
    files.sort()
    rows = []
    for fp in files:
        rows.append(process_file_raw(fp, task_config=task_config))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding='utf-8')
