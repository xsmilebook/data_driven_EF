import os
import pandas as pd
import numpy as np
from statistics import NormalDist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, 'data', 'EFNY', 'behavior_data', 'cibr_app_data')
TABLE_DIR = os.path.join(ROOT, 'data', 'EFNY', 'table')
TASK_CSV = os.path.join(TABLE_DIR, 'EFNY_task.csv')
OUT_CSV = os.path.join(TABLE_DIR, 'EFNY_metrics.csv')

KNOWN_METRICS = ['d_prime', 'SSRT', 'ACC', 'Reaction_Time', 'Constrast_ACC', 'Constrast_RT', 'Switch_Cost']
SKIP_TASKS = {'GNG', 'ZYST', 'FZSS'}

RT_MIN = 0.2
RT_MAX = 20.0

# 规范列名（所有 sheet 必须使用这些列名）
REQUIRED_COLUMNS = ['task', 'trial_index', 'subject_id', 'subject_name', 'item', 'answer', 'key', 'rt']

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

def to_numeric(series):
    s = pd.to_numeric(series, errors='coerce')
    return s

def clean_rt(rt):
    s = to_numeric(rt).dropna()
    s = s[(s >= RT_MIN) & (s <= RT_MAX)]
    if len(s) == 0:
        return s
    q1 = np.percentile(s, 25)
    q3 = np.percentile(s, 75)
    iqr = q3 - q1
    low = q1 - 3 * iqr
    high = q3 + 3 * iqr
    return s[(s >= low) & (s <= high)]

def determine_correct(df):
    if 'key' not in df.columns or 'answer' not in df.columns:
        return None
    a = df['answer'].astype(str).str.strip().str.lower()
    k = df['key'].astype(str).str.strip().str.lower()
    return k == a

def metric_acc(df):
    corr = determine_correct(df)
    if corr is None:
        return np.nan
    return float(corr.mean()) if len(corr) > 0 else np.nan

def metric_rt(df):
    if 'rt' not in df.columns:
        return np.nan
    corr = determine_correct(df)
    s = df['rt']
    if corr is not None:
        s = s[corr]
    s = clean_rt(s)
    return float(s.mean()) if len(s) > 0 else np.nan

def metric_dprime(df):
    if 'item' not in df.columns:
        return np.nan
    corr = determine_correct(df)
    if corr is None:
        return np.nan
    txt = df['item'].astype(str).str.lower()
    target_mask = txt.str.contains('target') | txt.str.contains('目标') | txt.str.contains('signal')
    noise_mask = txt.str.contains('non') | txt.str.contains('干扰') | txt.str.contains('noise')
    hits = ((target_mask) & (corr)).sum()
    signal_trials = int(target_mask.sum())
    false_alarms = ((noise_mask) & (~corr)).sum()
    noise_trials = int(noise_mask.sum())
    if signal_trials == 0 or noise_trials == 0:
        return np.nan
    H = (hits + 0.5) / (signal_trials + 1)
    FA = (false_alarms + 0.5) / (noise_trials + 1)
    nd = NormalDist()
    return float(nd.inv_cdf(H) - nd.inv_cdf(FA))

def metric_contrast_acc(df):
    if 'item' not in df.columns:
        return np.nan
    corr = determine_correct(df)
    if corr is None:
        return np.nan
    it = df['item'].astype(str).str.lower()
    incon_mask = it.str.contains('incongruent') | it.str.contains('不一致') | it.str.contains('incompatible') | it.str.contains('incon') | it.str.contains('ic')
    congr_mask = it.str.contains('congruent') | it.str.contains('一致') | it.str.contains('compatible') | it.str.contains('con') | it.str.contains('c')
    incon_acc = float(corr[incon_mask].mean()) if incon_mask.sum() > 0 else np.nan
    congr_acc = float(corr[congr_mask].mean()) if congr_mask.sum() > 0 else np.nan
    if np.isnan(incon_acc) or np.isnan(congr_acc):
        return np.nan
    return incon_acc - congr_acc

def metric_contrast_rt(df):
    if 'item' not in df.columns or 'rt' not in df.columns:
        return np.nan
    corr = determine_correct(df)
    s = df['rt']
    if corr is not None:
        s = s[corr]
    s = clean_rt(s)
    if len(s) == 0:
        return np.nan
    it = df.loc[s.index, 'item'].astype(str).str.lower()
    incon_idx = it.str.contains('incongruent') | it.str.contains('不一致') | it.str.contains('incompatible') | it.str.contains('incon') | it.str.contains('ic')
    congr_idx = it.str.contains('congruent') | it.str.contains('一致') | it.str.contains('compatible') | it.str.contains('con') | it.str.contains('c')
    incon_rt = clean_rt(df.loc[s.index[incon_idx], 'rt']) if incon_idx.sum() > 0 else pd.Series(dtype=float)
    congr_rt = clean_rt(df.loc[s.index[congr_idx], 'rt']) if congr_idx.sum() > 0 else pd.Series(dtype=float)
    if len(incon_rt) == 0 or len(congr_rt) == 0:
        return np.nan
    return float(incon_rt.mean() - congr_rt.mean())

def metric_ssrt(df):
    if 'item' not in df.columns or 'rt' not in df.columns:
        return np.nan
    it = df['item'].astype(str).str.lower()
    go_mask = it.str.contains('go') | it.str.contains('继续')
    stop_mask = it.str.contains('stop') | it.str.contains('停止')
    go_rt = clean_rt(df.loc[go_mask, 'rt'])
    ssd_ms = it.str.extract(r'(?:ssd|delay|stop_delay|停止信号延迟)[^0-9]*([0-9]+(?:\.[0-9]+)?)')[0].astype(float)
    ssd = ssd_ms[stop_mask].dropna()
    if len(ssd) > 0 and ssd.max() > 10:
        ssd = ssd / 1000.0
    if len(go_rt) == 0 or len(ssd) == 0:
        return np.nan
    if 'answer' not in df.columns or 'key' not in df.columns:
        return np.nan
    ans = df.loc[stop_mask, 'answer'].astype(str).str.strip().str.lower()
    key = df.loc[stop_mask, 'key'].astype(str).str.strip().str.lower()
    no_press_alias = ['no', 'none', '不按', '未按', '空', 'null', 'n/a']
    ans_norm = ans.where(~ans.isin(no_press_alias), 'no_press')
    key_norm = key.where(~key.isin(no_press_alias), 'no_press')
    inhibit_success = ans_norm == key_norm
    p_inhibit = float(inhibit_success.mean()) if len(inhibit_success) > 0 else np.nan
    if np.isnan(p_inhibit):
        return np.nan
    p_resp_stop = 1.0 - p_inhibit
    q = float(np.quantile(go_rt, min(max(p_resp_stop, 0.0), 1.0)))
    return q - float(ssd.mean())

def metric_switch_cost(df):
    if 'rt' not in df.columns or 'item' not in df.columns:
        return np.nan
    corr = determine_correct(df)
    s = df['rt']
    if corr is not None:
        s = s[corr]
    s = clean_rt(s)
    if len(s) == 0:
        return np.nan
    vals = df.loc[s.index, 'item'].astype(str).str.lower()
    sw_mask = vals.str.contains('switch') | vals.str.contains('切换')
    rep_mask = vals.str.contains('repeat') | vals.str.contains('重复') | vals.str.contains('noswitch')
    sw_rt = clean_rt(df.loc[s.index[sw_mask], 'rt']) if sw_mask.sum() > 0 else pd.Series(dtype=float)
    rep_rt = clean_rt(df.loc[s.index[rep_mask], 'rt']) if rep_mask.sum() > 0 else pd.Series(dtype=float)
    if len(sw_rt) == 0 or len(rep_rt) == 0:
        return np.nan
    return float(sw_rt.mean() - rep_rt.mean())

METRIC_FUNCS = {
    'ACC': metric_acc,
    'Reaction_Time': metric_rt,
    'd_prime': metric_dprime,
    'SSRT': metric_ssrt,
    'Constrast_ACC': metric_contrast_acc,
    'Constrast_RT': metric_contrast_rt,
    'Switch_Cost': metric_switch_cost,
}

def subject_code_from_filename(name):
    base = os.path.basename(name)
    return base.replace('_GameData.xlsx', '')

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
        # 若规范列缺失，则该任务的指标全部 NA
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        
        for m in metrics:
            func = METRIC_FUNCS.get(m)
            val = np.nan if len(missing) > 0 and m in ['ACC','Reaction_Time','d_prime','SSRT','Constrast_ACC','Constrast_RT','Switch_Cost'] else (func(df) if func is not None else np.nan)
            res[f'{task}_{m}'] = val
    return res

def main():
    tasks = read_task_config(TASK_CSV)
    files = []
    for fn in os.listdir(DATA_DIR):
        if fn.lower().endswith('.xlsx'):
            files.append(os.path.join(DATA_DIR, fn))
    files.sort()
    rows = []
    for fp in files:
        rows.append(process_file(fp, tasks))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

if __name__ == '__main__':
    main()