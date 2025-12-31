import os

import numpy as np
import pandas as pd

from .io import normalize_columns, subject_code_from_filename
from .metrics import get_raw_metrics


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


DEFAULT_TASK_CONFIG = {
    'oneback_number': {
        'type': 'nback',
        'n_back': 1,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'oneback_spatial': {
        'type': 'nback',
        'n_back': 1,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'oneback_emotion': {
        'type': 'nback',
        'n_back': 1,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'twoback_number': {
        'type': 'nback',
        'n_back': 2,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'twoback_spatial': {
        'type': 'nback',
        'n_back': 2,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'twoback_emotion': {
        'type': 'nback',
        'n_back': 2,
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Hit_Rate', 'FA_Rate', 'dprime'],
    },
    'FLANKER': {
        'type': 'conflict',
        'filter_rt': True,
        'rt_max': 2.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Congruent_ACC', 'Congruent_RT', 'Incongruent_ACC', 'Incongruent_RT',
                    'Contrast_RT', 'Contrast_ACC'],
    },
    'ColorStroop': {
        'type': 'conflict',
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Congruent_ACC', 'Congruent_RT', 'Incongruent_ACC', 'Incongruent_RT',
                    'Contrast_RT', 'Contrast_ACC'],
    },
    'EmotionStroop': {
        'type': 'conflict',
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Congruent_ACC', 'Congruent_RT', 'Incongruent_ACC', 'Incongruent_RT',
                    'Contrast_RT', 'Contrast_ACC'],
    },
    'DCCS': {
        'type': 'switch',
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'mixed_from': 22,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Repeat_ACC', 'Repeat_RT', 'Switch_ACC', 'Switch_RT',
                    'Switch_Cost_RT', 'Switch_Cost_ACC'],
    },
    'DT': {
        'type': 'switch',
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'mixed_from': 66,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Repeat_ACC', 'Repeat_RT', 'Switch_ACC', 'Switch_RT',
                    'Switch_Cost_RT', 'Switch_Cost_ACC'],
    },
    'EmotionSwitch': {
        'type': 'switch',
        'filter_rt': True,
        'rt_max': 2.5,
        'min_prop': 0.5,
        'mixed_from': 66,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD',
                    'Repeat_ACC', 'Repeat_RT', 'Switch_ACC', 'Switch_RT',
                    'Switch_Cost_RT', 'Switch_Cost_ACC'],
    },
    'SST': {
        'type': 'sst',
        'filter_rt': False,
        'rt_max': 2.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'SSRT', 'Mean_SSD', 'Stop_ACC', 'Go_RT_Mean', 'Go_RT_SD'],
    },
    'GNG': {
        'type': 'gonogo',
        'filter_rt': False,
        'rt_max': 3.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Go_ACC', 'NoGo_ACC', 'Go_RT_Mean', 'Go_RT_SD', 'dprime'],
    },
    'CPT': {
        'type': 'gonogo',
        'filter_rt': False,
        'rt_max': 3.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Go_ACC', 'NoGo_ACC', 'Go_RT_Mean', 'Go_RT_SD', 'dprime'],
    },
    'ZYST': {
        'type': 'zyst',
        'filter_rt': False,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'T0_ACC', 'T1_ACC', 'T1_given_T0_ACC', 'T0_RT', 'T1_RT'],
    },
    'FZSS': {
        'type': 'fzss',
        'filter_rt': True,
        'rt_max': 2.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Overall_ACC', 'Miss_Rate', 'FA_Rate', 'Correct_RT_Mean', 'Correct_RT_SD'],
    },
    'KT': {
        'type': 'kt',
        'filter_rt': False,
        'rt_max': 3.0,
        'min_prop': 0.5,
        'metrics': ['ACC', 'RT_Mean', 'RT_SD', 'Overall_ACC', 'Mean_RT'],
    },
}


def process_file_raw(path, task_config=None):
    task_config = task_config or DEFAULT_TASK_CONFIG
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


def run_raw(data_dir, out_csv, task_config=None):
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
