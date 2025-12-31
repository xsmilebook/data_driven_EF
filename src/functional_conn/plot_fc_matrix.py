import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets

from src.path_config import load_paths_config, resolve_dataset_roots

# command:
# Example:
# python -m src.functional_conn.plot_fc_matrix --file data/processed/avg_functional_conn_matrix/GroupAverage_Schaefer100_fisher_z.csv --out outputs/figures/functional_conn/GroupAverage_Schaefer100_fisher_z_yeo17.png --title "Group average fc (Schaefer 100)" --yeo17 --n-rois 100


def load_matrix(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception as e:
        print(f"Error loading matrix {path}: {e}", file=sys.stderr)
        sys.exit(1)

def get_yeo17_network_mapping(n_rois: int) -> tuple[np.ndarray, list[str]]:
    """
    Get Yeo17 network mapping for Schaefer atlas.
    Returns network labels for each ROI and network names.
    """
    try:
        # Fetch Schaefer atlas with Yeo17 networks
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=17, resolution_mm=2)
        
        # Get network labels for each ROI
        network_labels = atlas['labels']
        
        # Define network mapping based on Yeo 17-network nomenclature
        network_mapping = {
            'VisCent': 1, 'VisPeri': 2,  # Visual networks
            'SomMotA': 3, 'SomMotB': 4,  # Somatomotor networks  
            'DorsAttnA': 5, 'DorsAttnB': 6,  # Dorsal attention networks
            'SalVentAttnA': 7, 'SalVentAttnB': 8,  # Salience/ventral attention networks
            'LimbicA': 9, 'LimbicB': 10,  # Limbic networks
            'ContA': 11, 'ContB': 12, 'ContC': 13,  # Control networks
            'DefaultA': 14, 'DefaultB': 15, 'DefaultC': 16,  # Default mode networks
            'TempPar': 17  # Temporal-parietal network
        }
        
        network_names_dict = {
            1: 'Visual_Central', 2: 'Visual_Peripheral',
            3: 'Somatomotor_A', 4: 'Somatomotor_B',
            5: 'Dorsal_Attention_A', 6: 'Dorsal_Attention_B',
            7: 'Salience_Ventral_Attention_A', 8: 'Salience_Ventral_Attention_B',
            9: 'Limbic_A', 10: 'Limbic_B',
            11: 'Control_A', 12: 'Control_B', 13: 'Control_C',
            14: 'Default_Mode_A', 15: 'Default_Mode_B', 16: 'Default_Mode_C',
            17: 'Temporal_Parietal'
        }
        
        # Extract network numbers from labels
        network_numbers = []
        
        for label in network_labels:
            label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
            
            if label_str == 'Background':
                network_numbers.append(0)  # Background
                continue
                
            # Parse the label format: "17Networks_LH_VisCent_ExStr_1"
            parts = label_str.split('_')
            if len(parts) >= 3:
                # Extract network name (3rd part)
                network_name = parts[2]
                if network_name in network_mapping:
                    network_numbers.append(network_mapping[network_name])
                else:
                    print(f"Warning: Unknown network name: {network_name}")
                    network_numbers.append(0)  # Unknown network
            else:
                network_numbers.append(0)  # Unknown network
        
        # Remove background (0) and ensure we have exactly n_rois labels
        valid_indices = [i for i, num in enumerate(network_numbers) if num != 0]
        
        if len(valid_indices) >= n_rois:
            # Take only the first n_rois valid labels
            network_numbers = [network_numbers[i] for i in valid_indices[:n_rois]]
        else:
            print(f"Warning: Only found {len(valid_indices)} valid ROIs, expected {n_rois}")
            # Pad with existing ones
            network_numbers = [network_numbers[i] for i in valid_indices]
            while len(network_numbers) < n_rois:
                # Repeat existing network assignments
                if len(valid_indices) > 0:
                    network_numbers.append(network_numbers[0])
                else:
                    network_numbers.append(1)  # Default to network 1
        
        # Create network names list in order
        network_names = [network_names_dict[i] for i in range(1, 18)]
        
        return np.array(network_numbers), network_names
        
    except Exception as e:
        print(f"Error fetching Yeo17 atlas: {e}", file=sys.stderr)
        return None, None

def reorder_matrix_by_networks(matrix: np.ndarray, network_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Reorder matrix so that ROIs from the same network are grouped together.
    Returns reordered matrix, reordering indices, and network boundaries.
    """
    # Validate input dimensions
    if len(network_labels) != matrix.shape[0] or len(network_labels) != matrix.shape[1]:
        raise ValueError(f"Network labels length ({len(network_labels)}) must match matrix dimensions ({matrix.shape})")
    
    # Get sorting indices based on network labels
    sort_indices = np.argsort(network_labels)
    
    # Reorder the matrix
    reordered_matrix = matrix[sort_indices][:, sort_indices]
    
    # Find network boundaries
    unique_networks = np.unique(network_labels)
    if len(unique_networks) == 0:
        return reordered_matrix, sort_indices, []
    
    boundaries = []
    current_pos = 0
    
    # Sort unique networks to ensure consistent ordering
    sorted_networks = sorted(unique_networks)
    
    for network in sorted_networks[:-1]:  # Exclude the last network
        network_size = np.sum(network_labels == network)
        current_pos += network_size
        boundaries.append(current_pos)
    
    return reordered_matrix, sort_indices, boundaries

def plot_fc_matrix(mat: np.ndarray, out_path: Path, title: str = None, 
                   network_boundaries: list[int] = None, network_labels: list[str] = None):
    """
    Plots the FC matrix using nilearn.plotting.plot_matrix.
    If network_boundaries is provided, adds black lines to separate networks.
    """
    if title is None:
        title = "Functional Connectivity Matrix"

    # Create figure with appropriate size
    fig_size = max(10, len(mat) * 0.1)  # Scale figure size with matrix size
    fig = plt.figure(figsize=(fig_size, fig_size))
    
    # Determine vmin/vmax centered at 0
    max_val = np.nanmax(np.abs(mat))
    if max_val <= 1.0:
        vmax = 1.0
        vmin = -1.0
    else:
        vmax = max_val
        vmin = -max_val

    # Plot matrix
    display = plotting.plot_matrix(mat, figure=fig, title=title, cmap='RdBu_r', 
                                   vmax=vmax, vmin=vmin, tri='full', reorder=False,
                                   colorbar=True)
    
    # Get the actual axes from the figure
    ax = fig.gca()
    
    # Add network boundary lines if provided
    if network_boundaries:
        # Add vertical and horizontal lines at network boundaries
        for boundary in network_boundaries:
            # Vertical lines
            ax.axvline(x=boundary-0.5, color='black', linewidth=2, alpha=0.8)
            # Horizontal lines  
            ax.axhline(y=boundary-0.5, color='black', linewidth=2, alpha=0.8)
    
    # Add network labels if provided
    if network_labels and network_boundaries:
        # Calculate network positions for labels
        prev_boundary = 0
        network_positions = []
        network_names = []
        
        for i, boundary in enumerate(network_boundaries + [len(mat)]):
            # Calculate center position of this network
            center_pos = (prev_boundary + boundary) / 2 - 0.5
            network_positions.append(center_pos)
            if i < len(network_labels):
                network_names.append(network_labels[i])
            prev_boundary = boundary
        
        # Add network labels as text
        for pos, name in zip(network_positions, network_names):
            # Bottom labels
            ax.text(pos, -0.02, name, rotation=45, ha='center', va='top', 
                   fontsize=8, transform=ax.get_xaxis_transform())
            # Left labels  
            ax.text(-0.02, pos, name, rotation=0, ha='right', va='center',
                   fontsize=8, transform=ax.get_yaxis_transform())
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize Functional Connectivity Matrix")

    parser.add_argument("--file", required=True, help="Path to the input FC matrix (CSV)")
    parser.add_argument("--out", help="Path to output image file. If not provided, saves to default figures dir with same basename.")
    parser.add_argument("--title", help="Title of the plot")
    parser.add_argument("--yeo17", action="store_true", help="Map to Yeo17 networks and group ROIs by network")
    parser.add_argument("--n-rois", type=int, help="Number of ROIs in Schaefer atlas (needed for Yeo17 mapping)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", dest="paths_config", type=str, default="configs/paths.yaml")
    
    args = parser.parse_args()
    
    inp_path = Path(args.file)
    if not inp_path.exists():
        print(f"Error: Input file not found: {inp_path}", file=sys.stderr)
        sys.exit(1)
        
    mat = load_matrix(inp_path)
    
    # Determine number of ROIs from matrix size if not provided
    if args.n_rois is None:
        n_rois = mat.shape[0]
        print(f"Auto-detected {n_rois} ROIs from matrix size")
    else:
        n_rois = args.n_rois
    
    # Initialize variables for network mapping
    network_boundaries = None
    network_labels = None
    plot_title = args.title if args.title else inp_path.stem
    
    if args.yeo17:
        print("Mapping to Yeo17 networks...")
        network_numbers, network_names = get_yeo17_network_mapping(n_rois)
        
        if network_numbers is not None and len(network_numbers) > 0:
            print(f"Found {len(network_names)} networks: {network_names}")
            print(f"Network labels shape: {network_numbers.shape}")
            print(f"Unique network values: {sorted(set(network_numbers))}")
            print(f"Matrix shape: {mat.shape}")
            
            # Validate dimensions
            if len(network_numbers) != mat.shape[0]:
                print(f"Error: Network labels length ({len(network_numbers)}) doesn't match matrix size ({mat.shape[0]})")
                print("Using original matrix order.")
            else:
                try:
                    # Reorder matrix by networks
                    reordered_mat, sort_indices, boundaries = reorder_matrix_by_networks(mat, network_numbers)
                    
                    # Update matrix and boundaries
                    mat = reordered_mat
                    network_boundaries = boundaries
                    network_labels = network_names
                    
                    # Update title
                    plot_title = f"{plot_title} - Yeo17 Networks"
                    print(f"Matrix reordered by networks. Boundaries at positions: {boundaries}")
                except Exception as e:
                    print(f"Error reordering matrix: {e}")
                    print("Using original matrix order.")
        else:
            print("Warning: Could not fetch Yeo17 network mapping. Using original matrix order.")
    
    if args.out:
        out_path = Path(args.out)
    else:
        if not args.dataset:
            print("Missing --dataset when --out is not provided.", file=sys.stderr)
            sys.exit(1)
        repo_root = Path(__file__).resolve().parents[2]
        paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
        roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
        suffix = "_yeo17" if args.yeo17 else ""
        filename = inp_path.stem + suffix + ".png"
        out_path = roots["outputs_root"] / "figures" / "functional_conn" / filename
        
    print(f"Plotting matrix from {inp_path}...")
    plot_fc_matrix(mat, out_path, plot_title, network_boundaries, network_labels)
    print(f"Saved visualization to: {out_path}")

if __name__ == "__main__":
    main()
