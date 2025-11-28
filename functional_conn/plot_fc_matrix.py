import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

def load_matrix(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception as e:
        print(f"Error loading matrix {path}: {e}", file=sys.stderr)
        sys.exit(1)

def plot_fc_matrix(mat: np.ndarray, out_path: Path, title: str = None):
    """
    Plots the FC matrix using nilearn.plotting.plot_matrix.
    """
    if title is None:
        title = "Functional Connectivity Matrix"

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Determine vmin/vmax centered at 0
    # For correlation, usually [-1, 1]. For Fisher Z, can be larger.
    # We let nilearn handle it or force symmetric.
    max_val = np.nanmax(np.abs(mat))
    if max_val <= 1.0:
        vmax = 1.0
        vmin = -1.0
    else:
        vmax = max_val
        vmin = -max_val

    # Plot
    plotting.plot_matrix(mat, figure=fig, title=title, cmap='RdBu_r', 
                         vmax=vmax, vmin=vmin, tri='full', reorder=False,
                         colorbar=True)
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize Functional Connectivity Matrix")
    
    # Default paths
    root_dir = Path(__file__).resolve().parents[2]
    default_out_dir = root_dir / "data" / "EFNY" / "figures" / "functional_conn"
    
    parser.add_argument("--file", required=True, help="Path to the input FC matrix (CSV)")
    parser.add_argument("--out", help="Path to output image file. If not provided, saves to default figures dir with same basename.")
    parser.add_argument("--title", help="Title of the plot")
    
    args = parser.parse_args()
    
    inp_path = Path(args.file)
    if not inp_path.exists():
        print(f"Error: Input file not found: {inp_path}", file=sys.stderr)
        sys.exit(1)
        
    mat = load_matrix(inp_path)
    
    if args.out:
        out_path = Path(args.out)
    else:
        # Construct default output path
        # e.g. sub-001_Schaefer100_FC.csv -> sub-001_Schaefer100_FC.png
        filename = inp_path.stem + ".png"
        out_path = default_out_dir / filename
        
    if args.title:
        title = args.title
    else:
        title = inp_path.stem
        
    print(f"Plotting matrix from {inp_path}...")
    plot_fc_matrix(mat, out_path, title)
    print(f"Saved visualization to: {out_path}")

if __name__ == "__main__":
    main()
