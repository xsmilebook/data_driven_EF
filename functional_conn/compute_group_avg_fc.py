import argparse
import sys
from pathlib import Path
import numpy as np

# Try to import plotting function
try:
    from plot_fc_matrix import plot_fc_matrix
except ImportError:
    # If running as script from another dir, add current dir to path
    sys.path.append(str(Path(__file__).parent))
    from plot_fc_matrix import plot_fc_matrix

def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")

def save_matrix(mat: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, mat, delimiter=",", fmt="%.6f")

def main():
    parser = argparse.ArgumentParser(description="Compute Group Average FC Matrix")
    
    root_dir = Path(__file__).resolve().parents[2]
    default_in_dir = root_dir / "data" / "EFNY" / "functional_conn"
    default_sublist = root_dir / "data" / "EFNY" / "table" / "sublist" / "rest_valid_sublist.txt"
    default_fig_dir = root_dir / "data" / "EFNY" / "figures" / "functional_conn"
    
    parser.add_argument("--sublist", default=str(default_sublist), help="Path to subject list file")
    parser.add_argument("--in-dir", default=str(default_in_dir), help="Root directory containing FC matrices")
    parser.add_argument("--n-rois", type=int, default=100, help="Schaefer resolution (e.g. 100, 200, 400)")
    parser.add_argument("--out-name", default="GroupAverage", help="Base name for output file")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualization for the group average")
    parser.add_argument("--type", choices=["fc", "fisher_z"], default="fc", help="Type of matrix to look for (fc or fisher_z)")
    
    args = parser.parse_args()
    
    sublist_path = Path(args.sublist)
    in_dir = Path(args.in_dir)
    
    if not sublist_path.exists():
        print(f"Error: Subject list not found at {sublist_path}", file=sys.stderr)
        sys.exit(1)
        
    # Determine folder and suffix based on type
    # For FC: folder = Schaefer{N}, suffix = _Schaefer{N}_FC.csv
    # For Fisher Z: user might point to functional_conn_z/rest/Schaefer{N}. 
    # If in-dir is functional_conn, we assume structure Schaefer{N}.
    # If type is fisher_z, the filename usually has _FC_z.csv.
    
    folder = in_dir / f"Schaefer{args.n_rois}"
    if not folder.exists():
        # Try checking if there's a 'rest' subdir (legacy/other script output)
        folder_rest = in_dir / "rest" / f"Schaefer{args.n_rois}"
        if folder_rest.exists():
            folder = folder_rest
    
    if not folder.exists():
        print(f"Error: Input directory not found: {folder}", file=sys.stderr)
        sys.exit(1)
        
    suffix = f"_Schaefer{args.n_rois}_FC.csv"
    if args.type == "fisher_z":
        # Check if file pattern is different
        # fisher_z_fc.py outputs: {subject}_Schaefer{N}_FC_z.csv
        suffix = f"_Schaefer{args.n_rois}_FC_z.csv"

    subjects = []
    with open(sublist_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                subjects.append(s)
                
    print(f"Found {len(subjects)} subjects in list. Checking files in {folder}...")
    
    matrices = []
    valid_subs = []
    
    for sub in subjects:
        # File name pattern: {sub}{suffix}
        fpath = folder / f"{sub}{suffix}"
        if fpath.exists():
            try:
                mat = load_matrix(fpath)
                matrices.append(mat)
                valid_subs.append(sub)
            except Exception as e:
                print(f"Error loading {fpath}: {e}", file=sys.stderr)
        else:
            # Silent skip or warn? Warn is better.
            # print(f"Warning: Missing file for {sub}: {fpath}", file=sys.stderr)
            pass
            
    if not matrices:
        print("Error: No valid matrices found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Computing average from {len(matrices)} subjects...")
    
    # Stack and average
    stack = np.stack(matrices, axis=0)
    avg_mat = np.mean(stack, axis=0)
    
    # Output path
    # "store in same path" -> save in `folder`
    out_filename = f"{args.out_name}_Schaefer{args.n_rois}_{args.type}.csv"
    out_path = folder / out_filename
    
    save_matrix(avg_mat, out_path)
    print(f"Saved group average matrix to: {out_path}")
    
    if args.visualize:
        # Save figure to figures dir
        fig_name = f"{args.out_name}_Schaefer{args.n_rois}_{args.type}.png"
        fig_path = default_fig_dir / fig_name
        
        title = f"Group Average {args.type.upper()} Matrix (N={len(matrices)})"
        print(f"Generating visualization at {fig_path}...")
        plot_fc_matrix(avg_mat, fig_path, title)
        print("Visualization saved.")

if __name__ == "__main__":
    main()
