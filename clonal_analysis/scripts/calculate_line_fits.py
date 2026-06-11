#!/usr/bin/env python
"""
Pre-calculate best-fit lines for varying numbers of lowest loss points.
For each dataset (NT and Bach2), fit lines to subsets from 50 to 500 points.
Uses PCA to find the principal axis through the 3D points.
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA


def load_and_extract_lowest_points(data_file, n_points=1000):
    """
    Load loss landscape data and extract the n lowest loss points.

    Parameters
    ----------
    data_file : str
        Path to the NPZ file containing loss landscape data
    n_points : int
        Number of lowest loss points to extract

    Returns
    -------
    points : np.ndarray
        Array of shape (n_points, 4) with columns [p_self, p_split_one, p_split_two, loss]
    """
    # Load data
    data = np.load(data_file)
    p_self_values = data['p_self_values']
    p_split_one_values = data['p_split_one_values']
    p_split_two_values = data['p_split_two_values']
    loss_grid = data['loss_grid']
    data.close()

    # Create a list to store all valid points
    all_points = []

    # Iterate through the grid
    for i, p_self in enumerate(p_self_values):
        for j, p_split_one in enumerate(p_split_one_values):
            for k, p_split_two in enumerate(p_split_two_values):
                loss = loss_grid[i, j, k]
                # Check if loss is valid (not NaN) and parameters sum to <= 1
                if not np.isnan(loss) and (p_self + p_split_one + p_split_two) <= 1.0:
                    all_points.append([p_self, p_split_one, p_split_two, loss])

    # Convert to numpy array
    all_points = np.array(all_points)

    # Sort by loss (ascending) and take the lowest n_points
    sorted_indices = np.argsort(all_points[:, 3])
    lowest_points = all_points[sorted_indices[:min(n_points, len(all_points))]]

    return lowest_points


def fit_line_to_points(points):
    """
    Fit a line to the given 3D points using PCA.
    The line passes through the centroid and follows the first principal component.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 4) with columns [p_self, p_split_one, p_split_two, loss]

    Returns
    -------
    line_params : dict
        Dictionary containing:
        - 'centroid': Center point of the line [p_self, p_split_one, p_split_two]
        - 'direction': Unit vector for line direction
        - 'explained_variance': How much variance is explained by the line
    fit_quality : dict
        Dictionary with fit quality metrics
    """
    # Extract coordinates (ignore loss values)
    X = points[:, :3]  # p_self, p_split_one, p_split_two

    # Calculate centroid
    centroid = np.mean(X, axis=0)

    # Center the data
    X_centered = X - centroid

    # Apply PCA to find principal axis
    pca = PCA(n_components=1)
    pca.fit(X_centered)

    # Get the direction vector (first principal component)
    direction = pca.components_[0]

    # Ensure consistent orientation (positive p_self component)
    if direction[0] < 0:
        direction = -direction

    # Project points onto the line and calculate residuals
    projections = X_centered @ direction.reshape(-1, 1) @ direction.reshape(1, -1)
    residuals = X_centered - projections
    distances = np.linalg.norm(residuals, axis=1)

    # Calculate fit quality metrics
    fit_quality = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'max_distance': float(np.max(distances)),
        'explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
        'n_points': len(points)
    }

    line_params = {
        'centroid': centroid.tolist(),
        'direction': direction.tolist(),
        'explained_variance': float(pca.explained_variance_[0])
    }

    return line_params, fit_quality


def calculate_all_line_fits(data_file, dataset_name, output_path):
    """
    Calculate line fits for all subset sizes from 50 to 500 points.

    Parameters
    ----------
    data_file : str
        Path to the loss landscape NPZ file
    dataset_name : str
        Name of the dataset (for labeling)
    output_path : str
        Path to save the JSON output
    """
    print(f"Loading {dataset_name} data from {data_file}...")

    # Load the lowest 500 points (max we'll need)
    all_points = load_and_extract_lowest_points(data_file, n_points=500)
    print(f"Loaded {len(all_points)} lowest loss points")

    # Calculate line fits for different subset sizes
    results = {
        'dataset': dataset_name,
        'line_fits': {}
    }

    # Generate subset sizes: 50, 60, 70, ..., 500
    subset_sizes = list(range(50, min(501, len(all_points) + 1), 10))

    print(f"Calculating line fits for {len(subset_sizes)} different subset sizes...")
    for n in tqdm(subset_sizes, desc="Fitting lines"):
        if n > len(all_points):
            break

        subset = all_points[:n]
        line_params, fit_quality = fit_line_to_points(subset)

        # Create parametric equation string
        c = line_params['centroid']
        d = line_params['direction']
        param_equation = (f"p_self = {c[0]:.4f} + t*{d[0]:.4f}, "
                         f"p_split_one = {c[1]:.4f} + t*{d[1]:.4f}, "
                         f"p_split_two = {c[2]:.4f} + t*{d[2]:.4f}")

        results['line_fits'][str(n)] = {
            'n_points': n,
            'line_params': line_params,
            'parametric_equation': param_equation,
            'fit_quality': fit_quality,
            'loss_range': {
                'min': float(subset[0, 3]),
                'max': float(subset[-1, 3]),
                'mean': float(np.mean(subset[:, 3])),
                'median': float(np.median(subset[:, 3]))
            }
        }

    # Save results
    print(f"Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{dataset_name} Line Fit Summary:")
    print(f"  Subset sizes: {subset_sizes[0]} to {subset_sizes[-1]} points")
    print(f"  Total fits calculated: {len(results['line_fits'])}")

    # Show how line parameters change
    if len(results['line_fits']) > 0:
        first_n = str(subset_sizes[0])
        last_n = str(subset_sizes[-1])

        print(f"\n  Line equation for {first_n} points:")
        print(f"    {results['line_fits'][first_n]['parametric_equation']}")
        print(f"    Explained variance: {results['line_fits'][first_n]['line_params']['explained_variance']:.4f}")
        print(f"    Mean distance from line: {results['line_fits'][first_n]['fit_quality']['mean_distance']:.4f}")

        print(f"\n  Line equation for {last_n} points:")
        print(f"    {results['line_fits'][last_n]['parametric_equation']}")
        print(f"    Explained variance: {results['line_fits'][last_n]['line_params']['explained_variance']:.4f}")
        print(f"    Mean distance from line: {results['line_fits'][last_n]['fit_quality']['mean_distance']:.4f}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-calculate best-fit lines for varying numbers of lowest loss points"
    )
    parser.add_argument(
        "--nt-data",
        type=str,
        default="./clonal_analysis/intermediate_files/nt/grid_search_loss.npz",
        help="Path to NT loss landscape data file"
    )
    parser.add_argument(
        "--bach2-data",
        type=str,
        default="./clonal_analysis/intermediate_files/bach2/grid_search_loss.npz",
        help="Path to Bach2 loss landscape data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./clonal_analysis/intermediate_files",
        help="Directory to save output JSON files"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Calculate line fits for NT
    nt_output = os.path.join(args.output_dir, "line_fits_nt.json")
    calculate_all_line_fits(args.nt_data, "NT", nt_output)

    print("\n" + "="*60 + "\n")

    # Calculate line fits for Bach2
    bach2_output = os.path.join(args.output_dir, "line_fits_bach2.json")
    calculate_all_line_fits(args.bach2_data, "Bach2", bach2_output)

    print("\n✓ Successfully calculated all line fits!")
    print(f"  NT line fits saved to: {nt_output}")
    print(f"  Bach2 line fits saved to: {bach2_output}")


if __name__ == "__main__":
    main()
