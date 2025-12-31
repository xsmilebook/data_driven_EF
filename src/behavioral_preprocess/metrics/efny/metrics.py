from statistics import NormalDist

import numpy as np
import pandas as pd

from .preprocess import clean_rt, determine_correct, prepare_trials, to_numeric


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

    task_name = df['task'].iloc[0] if 'task' in df.columns and len(df) > 0 else ''

    if 'CPT' in str(task_name):
        ans = df['answer'].astype(str).str.lower()
        key = df['key'].astype(str).str.lower()
        target_mask = ans == 'true'
        noise_mask = ans == 'false'

        hits = ((target_mask) & (key == 'true')).sum()
        signal_trials = int(target_mask.sum())

        false_alarms = ((noise_mask) & (key == 'true')).sum()
        noise_trials = int(noise_mask.sum())
    else:
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

    task_name = df['task'].iloc[0] if 'task' in df.columns and len(df) > 0 else ''

    if 'FLANKER' in str(task_name).upper():
        congr_mask = it.str.endswith('ll') | it.str.endswith('rr')
        incon_mask = it.str.endswith('rl') | it.str.endswith('lr')
    elif 'COLORSTROOP' in str(task_name).upper():
        import re

        def check_stroop(s):
            m = re.search(r'pic_([a-z]+)_text_([a-z]+)', s)
            if m:
                return 'con' if m.group(1) == m.group(2) else 'incon'
            return None

        types = it.apply(check_stroop)
        congr_mask = types == 'con'
        incon_mask = types == 'incon'
    elif 'EMOTIONSTROOP' in str(task_name).upper() or 'EMOTIONSTOOP' in str(task_name).upper():
        import re

        def check_emotion_stroop(s):
            m = re.search(r'e(\d+)', s)
            if m:
                num = int(m.group(1))
                if 1 <= num <= 16:
                    return 'con'
                elif 17 <= num <= 32:
                    return 'incon'
            return None

        types = it.apply(check_emotion_stroop)
        congr_mask = types == 'con'
        incon_mask = types == 'incon'
    else:
        incon_mask = it.str.contains('incongruent') | it.str.contains('不一致') | it.str.contains(
            'incompatible') | it.str.contains('incon') | it.str.contains('ic')
        congr_mask = it.str.contains('congruent') | it.str.contains('一致') | it.str.contains(
            'compatible') | it.str.contains('con') | it.str.contains('c')

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

    df_s = df.loc[s.index]
    it = df_s['item'].astype(str).str.lower()
    task_name = df_s['task'].iloc[0] if 'task' in df_s.columns and len(df_s) > 0 else ''

    if 'FLANKER' in str(task_name).upper():
        congr_idx = it.str.endswith('ll') | it.str.endswith('rr')
        incon_idx = it.str.endswith('rl') | it.str.endswith('lr')
    elif 'COLORSTROOP' in str(task_name).upper():
        import re

        def check_stroop(s):
            m = re.search(r'pic_([a-z]+)_text_([a-z]+)', s)
            if m:
                return 'con' if m.group(1) == m.group(2) else 'incon'
            return None

        types = it.apply(check_stroop)
        congr_idx = types == 'con'
        incon_idx = types == 'incon'
    elif 'EMOTIONSTROOP' in str(task_name).upper() or 'EMOTIONSTOOP' in str(task_name).upper():
        import re

        def check_emotion_stroop(s):
            m = re.search(r'e(\d+)', s)
            if m:
                num = int(m.group(1))
                if 1 <= num <= 16:
                    return 'con'
                elif 17 <= num <= 32:
                    return 'incon'
            return None

        types = it.apply(check_emotion_stroop)
        congr_idx = types == 'con'
        incon_idx = types == 'incon'
    else:
        incon_idx = it.str.contains('incongruent') | it.str.contains('不一致') | it.str.contains(
            'incompatible') | it.str.contains('incon') | it.str.contains('ic')
        congr_idx = it.str.contains('congruent') | it.str.contains('一致') | it.str.contains(
            'compatible') | it.str.contains('con') | it.str.contains('c')

    incon_rt = clean_rt(df.loc[s.index[incon_idx], 'rt']) if incon_idx.sum() > 0 else pd.Series(dtype=float)
    congr_rt = clean_rt(df.loc[s.index[congr_idx], 'rt']) if congr_idx.sum() > 0 else pd.Series(dtype=float)
    if len(incon_rt) == 0 or len(congr_rt) == 0:
        return np.nan
    return float(incon_rt.mean() - congr_rt.mean())


def metric_ssrt(df):
    if 'SSRT' in df.columns:
        it_valid = False
        if 'item' in df.columns:
            it_str = df['item'].astype(str).str.lower()
            if it_str.str.contains('go').any() or it_str.str.contains('stop').any():
                it_valid = True

        if not it_valid:
            ssrt_col = df['SSRT'].astype(str)
            stop_mask = ssrt_col != '-'
            go_mask = ssrt_col == '-'

            go_rt = clean_rt(df.loc[go_mask, 'rt'])
            ssd = to_numeric(ssrt_col[stop_mask])

            if len(ssd) > 0 and ssd.max() > 10:
                ssd = ssd / 1000.0

            if len(go_rt) == 0 or len(ssd) == 0:
                return np.nan

            if 'key' not in df.columns:
                return np.nan

            key = df.loc[stop_mask, 'key'].astype(str).str.strip().str.lower()
            no_press_alias = ['no', 'none', '不按', '未按', '空', 'null', 'n/a', 'nan']

            inhibit_success = key.isin(no_press_alias)

            p_inhibit = float(inhibit_success.mean()) if len(inhibit_success) > 0 else np.nan
            if np.isnan(p_inhibit):
                return np.nan

            p_resp_stop = 1.0 - p_inhibit
            q = float(np.quantile(go_rt, min(max(p_resp_stop, 0.0), 1.0)))
            return q - float(ssd.mean())

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

    if 'trial_index' in df.columns:
        try:
            df['trial_index_num'] = pd.to_numeric(df['trial_index'], errors='coerce')
            df = df.sort_values('trial_index_num')
        except:
            pass

    corr = determine_correct(df)
    s = df['rt']
    if corr is not None:
        s = s[corr]
    s = clean_rt(s)
    if len(s) == 0:
        return np.nan

    df_s = df.loc[s.index]

    task_name = df_s['task'].iloc[0] if 'task' in df_s.columns and len(df_s) > 0 else ''

    if 'DCCS' in str(task_name).upper():
        full_rules = df['item'].astype(str).str.strip().str[-1]
        prev_rules = full_rules.shift(1)

        is_switch = (full_rules != prev_rules) & (prev_rules.notna())
        is_repeat = (full_rules == prev_rules) & (prev_rules.notna())

        sw_mask = is_switch.loc[s.index]
        rep_mask = is_repeat.loc[s.index]
    elif 'EMOTIONSWITCH' in str(task_name).upper():

        def get_rule(item_str):
            try:
                suffix = int(item_str.split('_')[-1])
                if 1 <= suffix <= 4:
                    return 'Emotion'
                elif 5 <= suffix <= 8:
                    return 'Gender'
            except:
                pass
            return None

        full_rules = df['item'].astype(str).apply(get_rule)
        prev_rules = full_rules.shift(1)

        is_switch = (full_rules != prev_rules) & (prev_rules.notna()) & (full_rules.notna())
        is_repeat = (full_rules == prev_rules) & (prev_rules.notna()) & (full_rules.notna())

        sw_mask = is_switch.loc[s.index]
        rep_mask = is_repeat.loc[s.index]
    elif 'DT' == str(task_name).upper() or 'DT ' in str(task_name).upper():

        def get_rule(ans_str):
            a = str(ans_str).lower().strip()
            if a in ['left', 'right']:
                return 'Horizontal'
            elif a in ['up', 'down']:
                return 'Vertical'
            return None

        if 'answer' in df.columns:
            full_rules = df['answer'].apply(get_rule)
            prev_rules = full_rules.shift(1)

            is_switch = (full_rules != prev_rules) & (prev_rules.notna()) & (full_rules.notna())
            is_repeat = (full_rules == prev_rules) & (prev_rules.notna()) & (full_rules.notna())

            sw_mask = is_switch.loc[s.index]
            rep_mask = is_repeat.loc[s.index]
        else:
            sw_mask = pd.Series(False, index=s.index)
            rep_mask = pd.Series(False, index=s.index)
    else:
        vals = df_s['item'].astype(str).str.lower()
        sw_mask = vals.str.contains('switch') | vals.str.contains('切换')
        rep_mask = vals.str.contains('repeat') | vals.str.contains('重复') | vals.str.contains('noswitch')

    sw_rt = clean_rt(df_s.loc[sw_mask, 'rt']) if sw_mask.sum() > 0 else pd.Series(dtype=float)
    rep_rt = clean_rt(df_s.loc[rep_mask, 'rt']) if rep_mask.sum() > 0 else pd.Series(dtype=float)

    if len(sw_rt) == 0 or len(rep_rt) == 0:
        return np.nan
    return float(sw_rt.mean() - rep_rt.mean())


def _clip_rate(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    eps = 0.5 / 100.0
    return float(min(max(v, eps), 1.0 - eps))


def compute_dprime(hit_rate, fa_rate):
    H = _clip_rate(hit_rate)
    FA = _clip_rate(fa_rate)
    if np.isnan(H) or np.isnan(FA):
        return np.nan
    nd = NormalDist()
    return float(nd.inv_cdf(H) - nd.inv_cdf(FA))


def _rt_stats_correct(df):
    if df is None or len(df) == 0 or 'rt' not in df.columns or 'correct_trial' not in df.columns:
        return (np.nan, np.nan)
    m = (df['correct_trial'] == True) & df['rt'].notna()
    s = pd.to_numeric(df.loc[m, 'rt'], errors='coerce').dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    mean_rt = float(s.mean())
    sd_rt = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return (mean_rt, sd_rt)


def _rt_stats_all(df):
    if df is None or len(df) == 0 or 'rt' not in df.columns:
        return (np.nan, np.nan)
    s = pd.to_numeric(df['rt'], errors='coerce').dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    mean_rt = float(s.mean())
    sd_rt = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return (mean_rt, sd_rt)


def _acc(df):
    if df is None or len(df) == 0 or 'correct_trial' not in df.columns:
        return np.nan
    s = df['correct_trial']
    s = s.dropna() if hasattr(s, 'dropna') else s
    return float(s.mean()) if len(s) > 0 else np.nan


def analyze_nback(df, cfg):
    n_back = int(cfg.get('n_back', 1))
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', True)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df']
    if 'item' not in d.columns:
        return None

    stim = d['item']
    lag = stim.shift(n_back)
    trial_type = pd.Series(np.nan, index=d.index, dtype='object')
    target_mask = stim.notna() & lag.notna() & (stim == lag)
    nontarget_mask = stim.notna() & ~target_mask
    trial_type.loc[nontarget_mask] = 'nontarget'
    trial_type.loc[target_mask] = 'target'

    d2 = d.copy()
    d2['trial_type'] = trial_type
    d2 = d2[d2['trial_type'].notna()]
    if len(d2) == 0:
        return None

    acc = _acc(d2)
    rt_mean, rt_sd = _rt_stats_all(d2)
    hit_rate = float(d2.loc[d2['trial_type'] == 'target', 'correct_trial'].mean()) if (d2['trial_type'] == 'target').any() else np.nan
    fa_rate = float((~d2.loc[d2['trial_type'] == 'nontarget', 'correct_trial']).mean()) if (d2['trial_type'] == 'nontarget').any() else np.nan

    return {
        'ACC': acc,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
        'Hit_Rate': hit_rate,
        'FA_Rate': fa_rate,
        'dprime': compute_dprime(hit_rate, fa_rate),
    }


def _conflict_condition(task_name, item_series):
    it = item_series.astype(str).str.strip()
    low = it.str.lower()
    if str(task_name).upper() == 'FLANKER':
        cong = low.str.endswith('ll') | low.str.endswith('rr')
        incong = low.str.endswith('lr') | low.str.endswith('rl')
    elif str(task_name).upper() == 'COLORSTROOP':
        parts = low.str.split('_', expand=True)
        if parts.shape[1] >= 4:
            pic_color = parts.iloc[:, 1]
            text_color = parts.iloc[:, 3]
            cong = pic_color == text_color
            incong = pic_color != text_color
        else:
            cong = pd.Series(False, index=low.index)
            incong = pd.Series(False, index=low.index)
    elif str(task_name).upper() == 'EMOTIONSTROOP':
        num = low.str.extract(r'(\d+)')[0]
        num = pd.to_numeric(num, errors='coerce')
        cong = (num % 4 == 0) & num.notna()
        incong = (num % 4 != 0) & num.notna()
    else:
        cong = low.str.contains('con') | low.str.contains('一致')
        incong = low.str.contains('incon') | low.str.contains('不一致')
    cond = pd.Series(np.nan, index=low.index, dtype='object')
    cond.loc[cong] = 'congruent'
    cond.loc[incong] = 'incongruent'
    return cond


def analyze_conflict(df, cfg, task_name):
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', True)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df']
    if 'item' not in d.columns:
        return None
    d = d.copy()
    d['cond'] = _conflict_condition(task_name, d['item'])
    d = d[d['cond'].notna()]
    if len(d) == 0:
        return None

    acc_all = _acc(d)
    rt_mean, rt_sd = _rt_stats_all(d)

    out = {
        'ACC': acc_all,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
    }

    for label, prefix in [('congruent', 'Congruent'), ('incongruent', 'Incongruent')]:
        sub = d[d['cond'] == label]
        out[f'{prefix}_ACC'] = _acc(sub)
        m, _ = _rt_stats_all(sub)
        out[f'{prefix}_RT'] = m

    out['Contrast_RT'] = out['Incongruent_RT'] - out['Congruent_RT'] if np.isfinite(out.get('Incongruent_RT', np.nan)) and np.isfinite(out.get('Congruent_RT', np.nan)) else np.nan
    out['Contrast_ACC'] = out['Incongruent_ACC'] - out['Congruent_ACC'] if np.isfinite(out.get('Incongruent_ACC', np.nan)) and np.isfinite(out.get('Congruent_ACC', np.nan)) else np.nan
    return out


def analyze_switch(df, cfg, task_name):
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', True)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df'].copy()
    if 'item' not in d.columns:
        return None

    if 'trial_index' in d.columns:
        d['_trial_index_num'] = pd.to_numeric(d['trial_index'], errors='coerce')
        d = d.sort_values('_trial_index_num')

    tn = str(task_name)
    rule = pd.Series(np.nan, index=d.index, dtype='object')
    if tn.upper() == 'DCCS':
        rule = d['item'].astype(str).str.slice(0, 1)
    elif tn.upper() == 'DT':
        rule = np.where(d['item'].astype(str).str.contains('[Tt]'), 'TN', 'CN')
        rule = pd.Series(rule, index=d.index, dtype='object')
    elif tn.upper() == 'EMOTIONSWITCH':
        num = d['item'].astype(str).str.extract(r'(\d+)')[0]
        num = pd.to_numeric(num, errors='coerce')
        rule = pd.Series(np.nan, index=d.index, dtype='object')
        em = num.notna() & num.between(1, 4)
        ge = num.notna() & num.between(5, 8)
        rule.loc[em] = 'emotion'
        rule.loc[ge] = 'gender'

    prev_rule = pd.Series(rule).shift(1)
    switch_type = pd.Series(np.nan, index=d.index, dtype='object')
    same = (rule == prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    diff = (rule != prev_rule) & prev_rule.notna() & pd.Series(rule).notna()
    switch_type.loc[same] = 'repeat'
    switch_type.loc[diff] = 'switch'

    d['switch_type'] = switch_type
    d = d[d['switch_type'].notna()]
    if len(d) == 0:
        return None

    mixed_from = cfg.get('mixed_from', None)
    if mixed_from is not None and '_trial_index_num' in d.columns:
        d = d[d['_trial_index_num'] >= float(mixed_from)]
    if len(d) == 0:
        return None

    acc_all = _acc(d)
    rt_mean, rt_sd = _rt_stats_all(d)

    out = {
        'ACC': acc_all,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
    }

    for label, prefix in [('repeat', 'Repeat'), ('switch', 'Switch')]:
        sub = d[d['switch_type'] == label]
        out[f'{prefix}_ACC'] = _acc(sub)
        m, _ = _rt_stats_all(sub)
        out[f'{prefix}_RT'] = m

    out['Switch_Cost_RT'] = out['Switch_RT'] - out['Repeat_RT'] if np.isfinite(out.get('Switch_RT', np.nan)) and np.isfinite(out.get('Repeat_RT', np.nan)) else np.nan
    out['Switch_Cost_ACC'] = out['Switch_ACC'] - out['Repeat_ACC'] if np.isfinite(out.get('Switch_ACC', np.nan)) and np.isfinite(out.get('Repeat_ACC', np.nan)) else np.nan
    return out


def analyze_sst(df, cfg):
    if df is None or len(df) == 0:
        return None

    d = df.copy()
    if len(d) >= 96:
        d = d.iloc[:96].copy()

    rt_var = cfg.get('rt_var', 'rt')
    if rt_var not in d.columns:
        return None
    d['rt'] = pd.to_numeric(d[rt_var], errors='coerce')

    ans_var = cfg.get('ans_var', 'answer')
    resp_var = cfg.get('resp_var', 'key')

    if ans_var in d.columns and resp_var in d.columns:
        ans = d[ans_var]
        resp = d[resp_var]
        correct_trial = ans.eq(resp)
        correct_trial = correct_trial.where(~(ans.isna() | resp.isna()), pd.NA)
        d['correct_trial'] = correct_trial
    else:
        d['correct_trial'] = pd.NA

    ssd_var = cfg.get('ssd_var', 'SSRT')
    if ssd_var not in d.columns:
        return {
            'ACC': np.nan,
            'RT_Mean': np.nan,
            'RT_SD': np.nan,
            'SSRT': np.nan,
            'Mean_SSD': np.nan,
            'Stop_ACC': np.nan,
            'Go_RT_Mean': np.nan,
            'Go_RT_SD': np.nan,
        }

    ssd = pd.to_numeric(d[ssd_var], errors='coerce')
    is_stop = ssd.notna()
    resp_series = d[resp_var] if resp_var in d.columns else pd.Series(pd.NA, index=d.index)

    correct = pd.Series(pd.NA, index=d.index, dtype='object')
    correct.loc[~is_stop] = d.loc[~is_stop, 'correct_trial']
    correct.loc[is_stop] = resp_series.loc[is_stop].isna()

    stop_trials = d.loc[is_stop]
    go_trials = d.loc[~is_stop]

    stop_acc = float(pd.to_numeric(correct.loc[is_stop], errors='coerce').mean()) if len(stop_trials) > 0 else np.nan
    mean_ssd = float(ssd.loc[is_stop].mean()) if len(stop_trials) > 0 else np.nan

    go_corr = go_trials.loc[correct.loc[~is_stop] == True].copy()
    if len(go_corr) == 0:
        return {
            'ACC': float(pd.to_numeric(correct, errors='coerce').mean()),
            'RT_Mean': np.nan,
            'RT_SD': np.nan,
            'SSRT': np.nan,
            'Mean_SSD': mean_ssd,
            'Stop_ACC': stop_acc,
            'Go_RT_Mean': np.nan,
            'Go_RT_SD': np.nan,
        }

    rt_max = cfg.get('rt_max', np.inf)
    min_prop = cfg.get('min_prop', 0.5)

    has_rt = go_corr['rt'].notna()
    go_corr_rt = go_corr.loc[has_rt].copy()
    n_rt_na = int((~has_rt).sum())
    n_rt_before = int(len(go_corr_rt))

    go_corr_rt = go_corr_rt[go_corr_rt['rt'] >= 0.2]
    if np.isfinite(rt_max):
        go_corr_rt = go_corr_rt[go_corr_rt['rt'] <= float(rt_max)]

    if len(go_corr_rt) > 0:
        m_rt = float(go_corr_rt['rt'].mean())
        s_rt = float(go_corr_rt['rt'].std(ddof=1)) if len(go_corr_rt) > 1 else np.nan
        if np.isfinite(s_rt) and s_rt > 0:
            z = (go_corr_rt['rt'] - m_rt) / s_rt
            go_corr_rt = go_corr_rt.loc[z.abs() <= 3].copy()

    n_rt_after = int(len(go_corr_rt))
    n_rt_deleted = int(n_rt_before - n_rt_after)
    n_problem = int(n_rt_deleted + n_rt_na)
    n_corr_all = int(len(go_corr))

    if n_corr_all > 0 and n_problem > n_corr_all * (1 - float(min_prop)):
        return {
            'ACC': float(pd.to_numeric(correct, errors='coerce').mean()),
            'RT_Mean': np.nan,
            'RT_SD': np.nan,
            'SSRT': np.nan,
            'Mean_SSD': mean_ssd,
            'Stop_ACC': stop_acc,
            'Go_RT_Mean': np.nan,
            'Go_RT_SD': np.nan,
        }

    if len(go_corr_rt) == 0:
        return {
            'ACC': float(pd.to_numeric(correct, errors='coerce').mean()),
            'RT_Mean': np.nan,
            'RT_SD': np.nan,
            'SSRT': np.nan,
            'Mean_SSD': mean_ssd,
            'Stop_ACC': stop_acc,
            'Go_RT_Mean': np.nan,
            'Go_RT_SD': np.nan,
        }

    ssrt = np.nan
    if np.isfinite(stop_acc) and (0 < stop_acc < 1) and np.isfinite(mean_ssd):
        sorted_go_rt = np.sort(go_corr_rt['rt'].values)
        p = 1.0 - float(stop_acc)
        n = len(sorted_go_rt)
        idx = int(np.floor(n * p))
        idx = max(1, min(n, idx))
        ssrt = float(sorted_go_rt[idx - 1]) - float(mean_ssd)

    mean_go_rt = float(go_corr_rt['rt'].mean())
    sd_go_rt = float(go_corr_rt['rt'].std(ddof=1)) if len(go_corr_rt) > 1 else np.nan

    return {
        'ACC': float(pd.to_numeric(correct, errors='coerce').mean()),
        'RT_Mean': mean_go_rt,
        'RT_SD': sd_go_rt,
        'SSRT': ssrt,
        'Mean_SSD': mean_ssd,
        'Stop_ACC': stop_acc,
        'Go_RT_Mean': mean_go_rt,
        'Go_RT_SD': sd_go_rt,
    }


def analyze_gng_cpt(df, cfg):
    if df is None or len(df) == 0:
        return None
    if 'answer' not in df.columns:
        return None

    d = df.copy()
    ans = d['answer'].astype(str).str.strip().str.lower()
    key = d['key'].astype(str).str.strip().str.lower() if 'key' in d.columns else pd.Series('', index=d.index)

    go_mask = ans.isin(['true', '1', 'yes'])
    nogo_mask = ans.isin(['false', '0', 'no'])
    if not (go_mask.any() or nogo_mask.any()):
        return None

    ans_raw = d['answer'] if 'answer' in d.columns else pd.Series(pd.NA, index=d.index)
    key_raw = d['key'] if 'key' in d.columns else pd.Series(pd.NA, index=d.index)
    correct_trial = ans_raw.eq(key_raw)
    correct_trial = correct_trial.where(~ans_raw.isna(), pd.NA)
    d['correct_trial'] = correct_trial

    d['trial_type'] = pd.NA
    d.loc[go_mask, 'trial_type'] = 'go'
    d.loc[nogo_mask, 'trial_type'] = 'nogo'

    go_trials_raw = d[d['trial_type'] == 'go']
    nogo_trials_all = d[d['trial_type'] == 'nogo']

    rt_var = cfg.get('rt_var', 'rt')
    rt_max = cfg.get('rt_max', np.inf)
    min_prop = cfg.get('min_prop', 0.5)

    if len(go_trials_raw) > 0 and rt_var in d.columns:
        go_trials_rt = go_trials_raw.copy()
        go_trials_rt['rt'] = pd.to_numeric(go_trials_rt[rt_var], errors='coerce')

        has_rt = go_trials_rt['rt'].notna()
        go_with_rt = go_trials_rt.loc[has_rt].copy()
        n_rt_na = int((~has_rt).sum())
        n_rt_before = int(len(go_with_rt))

        go_with_rt = go_with_rt[go_with_rt['rt'] >= 0.2]
        if np.isfinite(rt_max):
            go_with_rt = go_with_rt[go_with_rt['rt'] <= float(rt_max)]

        if len(go_with_rt) > 0:
            m_rt = float(go_with_rt['rt'].mean())
            s_rt = float(go_with_rt['rt'].std(ddof=1)) if len(go_with_rt) > 1 else np.nan
            if np.isfinite(s_rt) and s_rt > 0:
                z = (go_with_rt['rt'] - m_rt) / s_rt
                go_with_rt = go_with_rt.loc[z.abs() <= 3].copy()

        n_rt_after = int(len(go_with_rt))
        n_rt_deleted = int(n_rt_before - n_rt_after)
        n_go_all = int(len(go_trials_raw))
        n_problem = int(n_rt_deleted + n_rt_na)

        if n_go_all > 0 and n_problem > n_go_all * (1 - float(min_prop)):
            return None

        go_trials_valid = go_with_rt
    else:
        go_trials_valid = go_trials_raw.iloc[0:0].copy()

    go_acc = float(pd.to_numeric(go_trials_valid['correct_trial'], errors='coerce').mean()) if len(go_trials_valid) > 0 else np.nan
    nogo_acc = float(pd.to_numeric(nogo_trials_all['correct_trial'], errors='coerce').mean()) if len(nogo_trials_all) > 0 else np.nan
    acc_all = float(pd.to_numeric(d['correct_trial'], errors='coerce').mean()) if len(d) > 0 else np.nan

    go_rt_mean = float(pd.to_numeric(go_trials_valid['rt'], errors='coerce').mean()) if len(go_trials_valid) > 0 else np.nan
    go_rt_sd = float(pd.to_numeric(go_trials_valid['rt'], errors='coerce').std(ddof=1)) if len(go_trials_valid) > 1 else np.nan

    dprime = compute_dprime(go_acc, 1.0 - nogo_acc if np.isfinite(nogo_acc) else np.nan)

    return {
        'ACC': acc_all,
        'RT_Mean': go_rt_mean,
        'RT_SD': go_rt_sd,
        'Go_ACC': go_acc,
        'NoGo_ACC': nogo_acc,
        'Go_RT_Mean': go_rt_mean,
        'Go_RT_SD': go_rt_sd,
        'dprime': dprime,
    }


def analyze_zyst(df, cfg):
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', False)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df'].copy()
    if 'trial_index' not in d.columns:
        return None

    resp_var = cfg.get('resp_var', 'key')
    min_resp = int(cfg.get('min_resp', 128))
    if resp_var in d.columns:
        n_valid_resp = int(d[resp_var].notna().sum())
        if n_valid_resp < min_resp:
            return None

    mat = d['trial_index'].astype(str).str.extract(r'^(\d+)\s*-\s*(\d+)')
    d['trial'] = pd.to_numeric(mat[0], errors='coerce')
    d['subtrial'] = pd.to_numeric(mat[1], errors='coerce')
    d = d[d['trial'].notna() & d['subtrial'].notna()]
    if len(d) == 0:
        return None

    t0_list = []
    t1_list = []
    t1_given_t0_correct = []

    for trial_id in sorted(d['trial'].dropna().unique()):
        group = d[d['trial'] == trial_id]
        if not set([0, 1]).issubset(set(group['subtrial'])):
            continue

        t0 = group[group['subtrial'] == 0].iloc[0]
        t1 = group[group['subtrial'] == 1].iloc[0]

        t0_val = t0.get('correct_trial', pd.NA)
        t1_val = t1.get('correct_trial', pd.NA)
        t0_corr = 1 if isinstance(t0_val, (bool, np.bool_)) and t0_val else 0
        t1_corr = 1 if isinstance(t1_val, (bool, np.bool_)) and t1_val else 0

        t0_list.append(t0_corr)
        t1_list.append(t1_corr)
        if t0_corr == 1:
            t1_given_t0_correct.append(t1_corr)

    if len(t0_list) == 0 or len(t1_list) == 0:
        return None

    t0_acc = float(np.mean(t0_list)) if len(t0_list) > 0 else np.nan
    t1_acc = float(np.mean(t1_list)) if len(t1_list) > 0 else np.nan
    t1_acc_given_t0_correct = float(np.mean(t1_given_t0_correct)) if len(t1_given_t0_correct) > 0 else np.nan

    t0_rt_mean, _ = _rt_stats_all(d[d['subtrial'] == 0])
    t1_rt_mean, _ = _rt_stats_all(d[d['subtrial'] == 1])

    acc_all = float(pd.to_numeric(d.get('correct_trial', pd.Series([], dtype='float')).fillna(False), errors='coerce').mean()) if len(d) > 0 else np.nan
    rt_mean, rt_sd = _rt_stats_all(d)

    return {
        'ACC': acc_all,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
        'T0_ACC': t0_acc,
        'T1_ACC': t1_acc,
        'T1_given_T0_ACC': t1_acc_given_t0_correct,
        'T0_RT': t0_rt_mean,
        'T1_RT': t1_rt_mean,
    }


def analyze_fzss(df, cfg):
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', True)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df'].copy()
    if 'answer' not in d.columns or 'key' not in d.columns:
        return None

    ans = d['answer'].astype(str).str.strip().str.lower()
    key = d['key'].astype(str).str.strip().str.lower()
    correct = (ans == key)
    d['correct_trial'] = correct

    overall_acc = _acc(d)

    miss_base = ans == 'right'
    miss_trials = miss_base & (key.isna() | (key != 'right'))
    miss_rate = float(miss_trials.sum() / miss_base.sum()) if miss_base.sum() > 0 else np.nan

    fa_base = ans == 'left'
    fa_trials = fa_base & (key == 'right')
    fa_rate = float(fa_trials.sum() / fa_base.sum()) if fa_base.sum() > 0 else np.nan

    rt_mean, rt_sd = _rt_stats_correct(d)

    return {
        'ACC': overall_acc,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
        'Overall_ACC': overall_acc,
        'Miss_Rate': miss_rate,
        'FA_Rate': fa_rate,
        'Correct_RT_Mean': rt_mean,
        'Correct_RT_SD': rt_sd,
    }


def analyze_kt(df, cfg):
    prep = prepare_trials(
        df,
        filter_rt=bool(cfg.get('filter_rt', True)),
        rt_min=float(cfg.get('rt_min', 0.2)),
        rt_max=float(cfg.get('rt_max', 20.0)),
        min_prop=float(cfg.get('min_prop', 0.5)),
    )
    if not prep.get('ok', False):
        return None
    d = prep['df']
    acc_all = _acc(d)
    rt_mean, rt_sd = _rt_stats_all(d)
    return {
        'ACC': acc_all,
        'RT_Mean': rt_mean,
        'RT_SD': rt_sd,
        'Overall_ACC': acc_all,
        'Mean_RT': rt_mean,
    }


def get_raw_metrics(df, cfg, task_name):
    metrics = list(cfg.get('metrics', []))
    res = None
    t = cfg.get('type', None)
    try:
        if t == 'nback':
            res = analyze_nback(df, cfg)
        elif t == 'conflict':
            res = analyze_conflict(df, cfg, task_name=task_name)
        elif t == 'switch':
            res = analyze_switch(df, cfg, task_name=task_name)
        elif t == 'sst':
            res = analyze_sst(df, cfg)
        elif t == 'gonogo':
            res = analyze_gng_cpt(df, cfg)
        elif t == 'zyst':
            res = analyze_zyst(df, cfg)
        elif t == 'fzss':
            res = analyze_fzss(df, cfg)
        elif t == 'kt':
            res = analyze_kt(df, cfg)
    except Exception:
        res = None

    out = {m: np.nan for m in metrics}
    if isinstance(res, dict):
        for k, v in res.items():
            if k in out:
                out[k] = v
    return out
