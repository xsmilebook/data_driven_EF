import argparse
import sys
from pathlib import Path
import numpy as np

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots

def get_plot_function():
    try:
        from src.imaging_preprocess.plot_fc_matrix import plot_fc_matrix
        return plot_fc_matrix
    except Exception:
        return None

def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")

def save_matrix(mat: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, mat, delimiter=",", fmt="%.6f")

def _resolve_defaults(args) -> tuple[Path, Path, Path]:
    if not args.dataset:
        raise ValueError("Missing --dataset (required when defaults are used).")
    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(
        paths_cfg,
        dataset_config_path=args.dataset_config,
        repo_root=repo_root,
    )
    files_cfg = dataset_cfg.get("files", {})

    sublist_rel = files_cfg.get("rest_valid_sublist_file") or files_cfg.get("sublist_file")
    if not sublist_rel:
        raise ValueError("Missing files.rest_valid_sublist_file or files.sublist_file in dataset config.")

    in_dir = roots["interim_root"] / "functional_conn" / "rest"
    sublist_path = roots["processed_root"] / sublist_rel
    out_dir = roots["processed_root"] / "avg_functional_conn_matrix"
    fig_dir = roots["outputs_root"] / "figures" / "functional_conn"
    return in_dir, sublist_path, out_dir, fig_dir

def process_atlas(sublist_path: Path, atlas_folder: Path, out_dir: Path, fig_dir: Path, 
                  out_name: str, visualize: bool, atlas_name: str) -> None:
    """Process a single atlas folder"""
    
    # Determine matrix type and suffix based on folder contents
    if atlas_folder.name.endswith('_z') or '_FC_z.csv' in str(list(atlas_folder.glob('*.csv'))[0] if list(atlas_folder.glob('*.csv')) else ''):
        suffix = f"_{atlas_name}_FC_z.csv"
        matrix_type = "fisher_z"
    else:
        suffix = f"_{atlas_name}_FC.csv"
        matrix_type = "fc"
    
    subjects = []
    with open(sublist_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                subjects.append(s)
                
    print(f"Processing atlas {atlas_name}: Found {len(subjects)} subjects in list. Checking files in {atlas_folder}...")
    
    matrices = []
    valid_subs = []
    
    for sub in subjects:
        # File name pattern: {sub}{suffix}
        fpath = atlas_folder / f"{sub}{suffix}"
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
        print(f"Warning: No valid matrices found for atlas {atlas_name}.", file=sys.stderr)
        return
        
    print(f"Computing average from {len(matrices)} subjects for atlas {atlas_name}...")
    
    # Stack and average
    stack = np.stack(matrices, axis=0)
    avg_mat = np.mean(stack, axis=0)
    
    # Output path - save to specified output directory
    out_filename = f"{out_name}_{atlas_name}_{matrix_type}.csv"
    out_path = out_dir / out_filename
    
    save_matrix(avg_mat, out_path)
    print(f"Saved group average matrix to: {out_path}")
    
    if visualize:
        plot_fc_matrix = get_plot_function()
        if plot_fc_matrix:
            # Save figure to output directory instead of default figures dir
            fig_name = f"{out_name}_{atlas_name}_{matrix_type}.png"
            fig_path = fig_dir / fig_name
            
            title = f"Group Average {matrix_type.upper()} Matrix {atlas_name} (N={len(matrices)})"
            print(f"Generating visualization at {fig_path}...")
            plot_fc_matrix(avg_mat, fig_path, title)
            print("Visualization saved.")
        else:
            print("Warning: Could not import plot_fc_matrix. Skipping visualization.")


def main():
    parser = argparse.ArgumentParser(description="Compute Group Average FC Matrix")

    parser.add_argument("--sublist", default=None, help="Path to subject list file")
    parser.add_argument("--in-dir", default=None, help="Root directory containing FC matrices (e.g., rest or task folder)")
    parser.add_argument("--out-dir", default=None, help="Output directory for average matrices")
    parser.add_argument("--fig-dir", default=None, help="Output directory for figures")
    parser.add_argument("--n-rois", type=int, help="Specific Schaefer resolution to process (e.g. 100, 200, 400). If not specified, processes all found atlases.")
    parser.add_argument("--atlas", help="Specific atlas name to process (e.g., Schaefer100). If not specified, processes all found atlases.")
    parser.add_argument("--out-name", default="GroupAverage", help="Base name for output file")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualization for the group average")
    parser.add_argument("--condition", default="rest", help="Condition name (rest, task, etc.) - used for output naming")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--dataset-config", dest="dataset_config", type=str, default=None)
    
    args = parser.parse_args()

    if args.sublist is None or args.in_dir is None or args.out_dir is None or args.fig_dir is None:
        in_dir, sublist_path, out_dir, fig_dir = _resolve_defaults(args)
        args.in_dir = str(in_dir)
        args.sublist = str(sublist_path)
        args.out_dir = str(out_dir)
        args.fig_dir = str(fig_dir)

    sublist_path = Path(args.sublist)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    
    if not sublist_path.exists():
        print(f"Error: Subject list not found at {sublist_path}", file=sys.stderr)
        sys.exit(1)
        
    if not in_dir.exists():
        print(f"Error: Input directory not found at {in_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all atlas folders in the input directory
    atlas_folders = []
    
    if args.atlas:
        # Specific atlas requested
        atlas_folder = in_dir / args.atlas
        if atlas_folder.exists():
            atlas_folders.append((args.atlas, atlas_folder))
        else:
            print(f"Error: Requested atlas folder not found: {atlas_folder}", file=sys.stderr)
            sys.exit(1)
    elif args.n_rois:
        # Specific n_rois requested
        atlas_name = f"Schaefer{args.n_rois}"
        atlas_folder = in_dir / atlas_name
        if atlas_folder.exists():
            atlas_folders.append((atlas_name, atlas_folder))
        else:
            print(f"Error: Atlas folder not found: {atlas_folder}", file=sys.stderr)
            sys.exit(1)
    else:
        # Process all atlas folders
        for item in in_dir.iterdir():
            if item.is_dir() and item.name.startswith('Schaefer'):
                atlas_folders.append((item.name, item))
        
        if not atlas_folders:
            print(f"Error: No Schaefer atlas folders found in {in_dir}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Found {len(atlas_folders)} atlas folders to process: {[name for name, _ in atlas_folders]}")
    
    # Process each atlas
    for atlas_name, atlas_folder in atlas_folders:
        print(f"\nProcessing atlas: {atlas_name}")
        process_atlas(sublist_path, atlas_folder, out_dir, fig_dir, 
                     args.out_name, args.visualize, atlas_name)

if __name__ == "__main__":
    main()
