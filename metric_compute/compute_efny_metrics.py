import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT, 'data', 'EFNY', 'behavior_data', 'cibr_app_data')
TABLE_DIR = os.path.join(ROOT, 'data', 'EFNY', 'table', 'metrics')
OUT_CSV = os.path.join(TABLE_DIR, 'EFNY_beh_metrics.csv')

def main():
    from efny.main import run_raw

    run_raw(data_dir=DATA_DIR, out_csv=OUT_CSV)

if __name__ == '__main__':
    main()
