import argparse
import sys
from pathlib import Path
import numpy as np

def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")

def fisher_z(mat: np.ndarray) -> np.ndarray:
    m = np.clip(mat, -0.999999, 0.999999)
    z_mat = np.arctanh(m)
    # Set diagonal elements to 0
    np.fill_diagonal(z_mat, 0)
    return z_mat

def save_matrix(mat: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, mat, delimiter=",", fmt="%.6f")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "functional_conn" / "rest"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "EFNY" / "functional_conn_z" / "rest"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--n-rois", type=int, default=100)
    args = parser.parse_args()

    subject = args.subject.strip()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    inp_folder = in_dir / f"Schaefer{args.n_rois}"
    inp_file = inp_folder / f"{subject}_Schaefer{args.n_rois}_FC.csv"
    if not inp_file.exists():
        print(f"Missing input: {inp_file}", file=sys.stderr)
        return

    mat = load_matrix(inp_file)
    z = fisher_z(mat)

    out_folder = out_dir / f"Schaefer{args.n_rois}"
    out_file = out_folder / f"{subject}_Schaefer{args.n_rois}_FC_z.csv"
    save_matrix(z, out_file)
    print(f"saved: {out_file}")

if __name__ == "__main__":
    main()

