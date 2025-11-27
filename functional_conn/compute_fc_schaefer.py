import argparse
import sys
from pathlib import Path
import numpy as np

## command
# python d:\code\data_driven_EF\src\FC_compute\compute_fc_schaefer100.py --subject sub-THU20250819728LZQ --out d:\code\data_driven_EF\data\EFNY\functional_conn\Schaefer100_FC.csv 

def find_runs(func_dir: Path, run_ids: list[int]) -> list[Path]:
    files = []
    for i in run_ids:
        pattern = f"*task-rest_run-{i}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
        m = list(func_dir.glob(pattern))
        if len(m) > 0:
            files.append(m[0])
    return files

def get_valid_runs(subject: str, qc_path: Path) -> list[int]:
    import csv
    valid_runs = []
    if not qc_path.exists():
        print(f"Warning: QC file not found at {qc_path}", file=sys.stderr)
        return []
        
    with open(qc_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['subid'] == subject:
                for i in range(1, 5):
                    col_name = f'rest{i}_valid'
                    # Check if column exists and value is '1'
                    if col_name in row and row[col_name] == '1':
                        valid_runs.append(i)
                return valid_runs
    return []

def is_valid_subject(subject: str, list_path: Path) -> bool:
    if not list_path.exists():
        print(f"Warning: Subject list file not found at {list_path}", file=sys.stderr)
        return False
        
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == subject:
                return True
    return False

def load_concat_img(run_files: list[Path]):
    from nilearn import image
    return image.concat_imgs([str(p) for p in run_files])

def compute_fc(concat_img, n_rois: int) -> np.ndarray:
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7, resolution_mm=2)
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
    ts = masker.fit_transform(concat_img)
    corr = np.corrcoef(ts, rowvar=False)
    return corr

def save_matrix(mat: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, mat, delimiter=",", fmt="%.6f")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcpd-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "MRI_data" / "xcpd_rest"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--n-rois", type=int, default=100, help="Number of Schaefer ROIs (e.g. 100, 200, 400)")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "functional_conn"))
    parser.add_argument("--qc-file", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "table" / "qc" / "rest_fd_summary.csv"))
    parser.add_argument("--valid-list", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "table" / "sublist" / "rest_valid_sublist.txt"))
    args = parser.parse_args()
    
    # 1. Check if subject is in the valid list
    valid_list_path = Path(args.valid_list)
    if not is_valid_subject(args.subject, valid_list_path):
        print(f"Skipping {args.subject}: Not in valid subject list.", file=sys.stderr)
        return

    # 2. Get valid runs from QC file
    qc_path = Path(args.qc_file)
    valid_run_ids = get_valid_runs(args.subject, qc_path)
    if not valid_run_ids:
        print(f"Skipping {args.subject}: No valid runs found in QC file (or subject not found in QC).", file=sys.stderr)
        return

    base = Path(args.xcpd_dir)
    sub = base / args.subject / "func"
    if not sub.exists():
        print(f"Not found: {sub}", file=sys.stderr)
        return
        
    runs = find_runs(sub, valid_run_ids)
    if len(runs) == 0:
        print(f"No corresponding run files found for {args.subject} (expected runs: {valid_run_ids})", file=sys.stderr)
        return
        
    print(f"Processing {args.subject} with runs: {valid_run_ids} -> Found {len(runs)} files.")
    
    try:
        img = load_concat_img(runs)
        mat = compute_fc(img, args.n_rois)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return
    
    out_base = Path(args.out_dir)
    out_folder = out_base / f"Schaefer{args.n_rois}"
    out_filename = f"{args.subject}_Schaefer{args.n_rois}_FC.csv"
    out_path = out_folder / out_filename
    
    save_matrix(mat, out_path)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()