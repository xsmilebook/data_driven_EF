import argparse
import sys
from pathlib import Path
import numpy as np

## command
# python d:\code\data_driven_EF\src\FC_compute\compute_fc_schaefer100.py --subject sub-THU20250819728LZQ --out d:\code\data_driven_EF\data\EFNY\functional_conn\Schaefer100_FC.csv 

def find_runs(func_dir: Path) -> list[Path]:
    files = []
    for i in range(1, 5):
        pattern = f"*task-rest_run-{i}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
        m = list(func_dir.glob(pattern))
        if len(m) > 0:
            files.append(m[0])
    return files

def load_concat_img(run_files: list[Path]):
    from nilearn import image
    return image.concat_imgs([str(p) for p in run_files])

def compute_fc(concat_img) -> np.ndarray:
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
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
    parser.add_argument("--out", default="Schaefer100_FC.csv")
    args = parser.parse_args()
    base = Path(args.xcpd_dir)
    sub = base / args.subject / "func"
    if not sub.exists():
        print(f"Not found: {sub}", file=sys.stderr)
        return
    runs = find_runs(sub)
    if len(runs) == 0:
        print("No runs found", file=sys.stderr)
        return
    try:
        img = load_concat_img(runs)
        mat = compute_fc(img)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = sub / out_path.name
    save_matrix(mat, out_path)
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()