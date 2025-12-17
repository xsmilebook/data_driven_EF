from statistics import NormalDist

import numpy as np
import pandas as pd

from .preprocess import clean_rt, determine_correct, to_numeric


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

