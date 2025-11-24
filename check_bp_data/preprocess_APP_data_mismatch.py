import os
import re
import pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
TABLE_XLSX = os.path.join(ROOT, 'data', 'EFNY', 'table', 'demo', 'bp_midterm_all_sites_demo.xlsx')
NEW_TABLE_XLSX = os.path.join(ROOT, 'data', 'EFNY', 'table', 'demo', 'bp_midterm_all_sites_demo_with_app.xlsx')
APP_DIR = os.path.join(ROOT, 'data', 'EFNY', 'behavior_data', 'all_sites_app_data_251124')

def extract_id_from_subj(s: str) -> str:
    if s is None:
        return ''
    t = str(s).strip()
    t = re.sub(r'[A-Za-z]+$', '', t)
    digits = ''.join(re.findall(r'\d', t))
    if not digits:
        return ''
    k = digits[-3:] if len(digits) >= 3 else digits
    return k

def collect_site_ids(site: str) -> set[int]:
    site_dir = os.path.join(APP_DIR, site)
    ids = set()
    if not os.path.isdir(site_dir):
        return ids
    for fn in os.listdir(site_dir):
        if not fn.lower().endswith('.xlsx'):
            continue
        parts = fn.split('_')
        if len(parts) < 3:
            continue
        seg = parts[2]
        m = re.search(r'(\d+)', seg)
        if not m:
            continue
        try:
            x = int(m.group(1))
        except Exception:
            continue
        ids.add(x)
    return ids

def update_sheet(df: pd.DataFrame, site_ids: set[int]) -> pd.DataFrame:
    df = df.copy()
    col = "subj_ID"
    if col is None:
        df['ID'] = ''
        df['APP_data'] = '0'
        return df
    ids = [extract_id_from_subj(v) for v in df[col]]
    df['ID'] = ids
    has = []
    for s in ids:
        if not s:
            has.append('0')
            continue
        try:
            v = int(s)
        except Exception:
            has.append('0')
            continue
        has.append('1' if v in site_ids else '0')
    df['APP_data'] = has
    return df

def main() -> None:
    xl = pd.ExcelFile(TABLE_XLSX)
    sheets = {name: xl.parse(name, dtype='object') for name in xl.sheet_names}
    thu_ids = collect_site_ids('THU')
    bnu_ids = collect_site_ids('BNU')
    xy_ids = collect_site_ids('XY')
    out_sheets = {}
    for name, df in sheets.items():
        if name.upper() in ('THU', 'CIBR'):
            out_sheets[name] = update_sheet(df, thu_ids)
        elif name.upper() == 'BNU':
            out_sheets[name] = update_sheet(df, bnu_ids)
        elif name.upper() == 'XY':
            out_sheets[name] = update_sheet(df, xy_ids)
        else:
            out_sheets[name] = df
    with pd.ExcelWriter(NEW_TABLE_XLSX, engine='openpyxl', mode='w') as writer:
        for name, df in out_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

if __name__ == '__main__':
    main()
