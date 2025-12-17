import os

import pandas as pd

from .registry import KNOWN_METRICS


def read_task_config(path):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, engine='python', on_bad_lines='warn')
    df.columns = [c.strip() for c in df.columns]
    df = df[[c for c in df.columns if c]]
    rows = []
    for _, row in df.iterrows():
        task = str(row.get('Task', '')).strip()
        if not task:
            continue
        metrics = []
        for m in KNOWN_METRICS:
            v = str(row.get(m, '')).strip().lower()
            if v in ('true', '1', 'yes'):
                metrics.append(m)
        rows.append({'Task': task, 'metrics': metrics})
    return rows


def normalize_columns(df):
    return df.rename(columns={
        '任务': 'task',
        '游戏序号': 'trial_index',
        '被试编号（用户账号）': 'subject_id',
        '被试姓名（真实姓名)': 'subject_name',
        '正式阶段刺激图片/Item名': 'item',
        '正式阶段正确答案': 'answer',
        '正式阶段被试按键': 'key',
        '绝对时间(待定)': 'abs_time',
        '相对时间(秒)': 'rt'
    })


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def subject_code_from_filename(name):
    base = os.path.basename(name)
    return base.replace('_GameData.xlsx', '')

