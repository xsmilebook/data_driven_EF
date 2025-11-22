import os
import re
import pandas as pd

BASE_DEMO_DIR = r"D:\code\data_driven_EF\data\EFNY\table\demo"

def _excel_path():
    return os.path.join(BASE_DEMO_DIR, 'bp_midterm_all_sites_demo.xlsx')

def _read_psych(csv_name):
    p = os.path.join(BASE_DEMO_DIR, csv_name)
    return pd.read_csv(p)

def _coerce_bool(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(['true', '1', 't'])
    return df

def _is_numeric(v):
    if pd.isna(v):
        return False
    s = str(v).strip()
    if s == '' or s.lower() == 'na':
        return False
    n = pd.to_numeric(s, errors='coerce')
    return not pd.isna(n)

def _has_any_imaging(row):
    return (_is_numeric(row.get('nback')) or _is_numeric(row.get('sst')) or _is_numeric(row.get('switch')))

def _thu_xy_psych_key(subid):
    return 'sub-' + subid.replace('_', '')

def _efi_psych_id(subid):
    parts = str(subid).split('_')
    if len(parts) >= 2:
        return parts[-2]
    return None

def _efi_excel_id(subj_id):
    if pd.isna(subj_id):
        return None
    s = str(subj_id).strip()
    s = re.sub(r'[A-Za-z]+$', '', s)
    m = re.search(r'(\d{3})$', s)
    return m.group(1) if m else None

def _process_thu():
    psych = _read_psych('psych_files_statistics_THU.csv')
    psych = _coerce_bool(psych, ['nback_complete', 'SST_complete', 'switch_complete'])
    excel = pd.read_excel(_excel_path(), sheet_name='CIBR')
    behavioral = set()
    for _, r in psych.iterrows():
        has_beh = bool(r.get('nback_complete')) or bool(r.get('SST_complete')) or bool(r.get('switch_complete'))
        if has_beh:
            behavioral.add(_thu_xy_psych_key(r.get('subid')))
    imaging = set()
    excel = excel.rename(columns={c: c.strip() for c in excel.columns})
    for _, r in excel.iterrows():
        if _has_any_imaging(r):
            sid = r.get('subj_ID')
            if not pd.isna(sid):
                imaging.add(str(sid).strip())
    missing = imaging - behavioral
    rows = []
    for _, r in excel.iterrows():
        sid = str(r.get('subj_ID')).strip() if not pd.isna(r.get('subj_ID')) else None
        if sid in missing:
            rows.append({
                'dataset': 'THU',
                'excel_subj_ID': sid,
                'psych_subid': None
            })
    return pd.DataFrame(rows)

def _process_xy():
    psych = _read_psych('psych_files_statistics_XY.csv')
    psych = _coerce_bool(psych, ['nback_complete', 'SST_complete', 'switch_complete'])
    excel = pd.read_excel(_excel_path(), sheet_name='XY')
    behavioral = set()
    for _, r in psych.iterrows():
        has_beh = bool(r.get('nback_complete')) or bool(r.get('SST_complete')) or bool(r.get('switch_complete'))
        if has_beh:
            behavioral.add(_thu_xy_psych_key(r.get('subid')))
    imaging = set()
    excel = excel.rename(columns={c: c.strip() for c in excel.columns})
    for _, r in excel.iterrows():
        if _has_any_imaging(r):
            sid = r.get('subj_ID')
            if not pd.isna(sid):
                imaging.add(str(sid).strip())
    missing = imaging - behavioral
    rows = []
    for _, r in excel.iterrows():
        sid = str(r.get('subj_ID')).strip() if not pd.isna(r.get('subj_ID')) else None
        if sid in missing:
            rows.append({
                'dataset': 'XY',
                'excel_subj_ID': sid,
                'psych_subid': None
            })
    return pd.DataFrame(rows)

def _process_efi():
    psych = _read_psych('psych_files_statistics_EFI.csv')
    psych = _coerce_bool(psych, ['nback_complete', 'SST_complete', 'switch_complete'])
    excel = pd.read_excel(_excel_path(), sheet_name='BNU')
    behavioral = set()
    for _, r in psych.iterrows():
        has_beh = bool(r.get('nback_complete')) or bool(r.get('SST_complete')) or bool(r.get('switch_complete'))
        if has_beh:
            pid = _efi_psych_id(r.get('subid'))
            if pid:
                behavioral.add(pid)
    imaging = set()
    excel = excel.rename(columns={c: c.strip() for c in excel.columns})
    for _, r in excel.iterrows():
        if _has_any_imaging(r):
            eid = _efi_excel_id(r.get('subj_ID'))
            if eid:
                imaging.add(eid)
    missing_ids = imaging - behavioral
    rows = []
    for _, r in excel.iterrows():
        eid = _efi_excel_id(r.get('subj_ID'))
        if eid in missing_ids:
            rows.append({
                'dataset': 'EFI',
                'excel_subj_ID': str(r.get('subj_ID')).strip() if not pd.isna(r.get('subj_ID')) else None,
                'psych_subid': None,
                'id': eid
            })
    return pd.DataFrame(rows)

def main():
    out_dir = BASE_DEMO_DIR
    thu_df = _process_thu()
    xy_df = _process_xy()
    efi_df = _process_efi()
    if not thu_df.empty:
        thu_df.to_csv(os.path.join(out_dir, 'THU_imaging_no_behavioral.csv'), index=False)
    if not xy_df.empty:
        xy_df.to_csv(os.path.join(out_dir, 'XY_imaging_no_behavioral.csv'), index=False)
    if not efi_df.empty:
        efi_df.to_csv(os.path.join(out_dir, 'EFI_imaging_no_behavioral.csv'), index=False)
    all_df = pd.concat([df for df in [thu_df, xy_df, efi_df] if not df.empty], ignore_index=True) if (not thu_df.empty or not xy_df.empty or not efi_df.empty) else pd.DataFrame()
    if not all_df.empty:
        all_df.to_csv(os.path.join(out_dir, 'all_datasets_imaging_no_behavioral.csv'), index=False)

if __name__ == '__main__':
    main()