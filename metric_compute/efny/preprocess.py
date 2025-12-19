import numpy as np
import pandas as pd

RT_MIN = 0.2
RT_MAX = 20.0


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


def _safe_mean(x):
    s = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    return float(s.mean()) if len(s) > 0 else np.nan


def _safe_sd(x):
    s = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else np.nan


def prepare_trials(df,
                   rt_col='rt',
                   answer_col='answer',
                   key_col='key',
                   compute_correct=True,
                   filter_rt=True,
                   rt_min=RT_MIN,
                   rt_max=RT_MAX,
                   min_prop=0.5):
    n_raw = int(len(df))
    if df is None:
        return {'ok': False, 'df': pd.DataFrame(), 'n_raw': 0, 'n_kept': 0}

    out = df.copy()

    if 'correct_trial' not in out.columns:
        out['correct_trial'] = pd.array([pd.NA] * len(out), dtype='boolean')
        if compute_correct and answer_col in out.columns and key_col in out.columns:
            a = out[answer_col].astype(str).str.strip()
            k = out[key_col].astype(str).str.strip()
            out['correct_trial'] = pd.array((a == k).values, dtype='boolean')
            out.loc[out[answer_col].isna(), 'correct_trial'] = pd.NA

    if rt_col in out.columns:
        out['rt'] = pd.to_numeric(out[rt_col], errors='coerce')
    else:
        if filter_rt:
            return {'ok': False, 'df': out, 'n_raw': n_raw, 'n_kept': int(len(out))}
        out['rt'] = np.nan

    if not filter_rt:
        return {'ok': True, 'df': out, 'n_raw': n_raw, 'n_kept': int(len(out))}

    has_rt = out['rt'].notna()
    df_rt = out.loc[has_rt].copy()
    df_no_rt = out.loc[~has_rt].copy()
    n_rt_na = int((~has_rt).sum())
    n_rt_before = int(len(df_rt))

    if n_rt_before > 0:
        df_rt = df_rt[(df_rt['rt'] >= float(rt_min)) & (df_rt['rt'] <= float(rt_max))]

    if len(df_rt) > 0:
        m = _safe_mean(df_rt['rt'])
        s = _safe_sd(df_rt['rt'])
        if np.isfinite(s) and s > 0:
            z = (df_rt['rt'] - m) / s
            df_rt = df_rt.loc[z.abs() <= 3].copy()

    n_rt_after = int(len(df_rt))
    n_rt_deleted = int(n_rt_before - n_rt_after)

    n_problem = int(n_rt_deleted + n_rt_na)
    if n_raw > 0 and (n_problem > n_raw * (1 - float(min_prop))):
        out_merged = pd.concat([df_rt, df_no_rt], axis=0)
        return {'ok': False, 'df': out_merged, 'n_raw': n_raw, 'n_kept': int(len(out_merged))}

    out_merged = pd.concat([df_rt, df_no_rt], axis=0)
    return {'ok': True, 'df': out_merged, 'n_raw': n_raw, 'n_kept': int(len(out_merged))}

