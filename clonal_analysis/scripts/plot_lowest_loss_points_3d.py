#!/usr/bin/env python
"""
Create an interactive 3D scatter plot showing the lowest loss points for Bach2 and NT datasets.
Features a slider to control the number of points displayed (50-1000) with corresponding best-fit lines.
Dynamically calculates best-fit lines using parallel processing.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from multiprocessing import Pool
from sklearn.decomposition import PCA
from functools import partial
from pathlib import Path


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

    line_params = {
        'centroid': centroid.tolist(),
        'direction': direction.tolist(),
        'explained_variance': float(pca.explained_variance_[0])
    }

    return line_params


def fit_line_with_metadata(points_and_n):
    """
    Wrapper function for parallel processing that includes metadata.

    Parameters
    ----------
    points_and_n : tuple
        (points, n) where points is the subset and n is the subset size

    Returns
    -------
    dict : Line fit result with metadata
    """
    points, n = points_and_n
    line_params = fit_line_to_points(points)

    # Create parametric equation string
    c = line_params['centroid']
    d = line_params['direction']
    param_equation = (f"p_sym = {c[0]:.4f} + t*{d[0]:.4f}, "
                     f"p_asym = {c[1]:.4f} + t*{d[1]:.4f}, "
                     f"p_dif = {c[2]:.4f} + t*{d[2]:.4f}")

    return {
        'n_points': n,
        'line_params': line_params,
        'parametric_equation': param_equation
    }


def calculate_line_fits_parallel(points, subset_sizes, n_cpus=32):
    """
    Calculate line fits in parallel for different subset sizes.

    Parameters
    ----------
    points : np.ndarray
        Array of lowest loss points
    subset_sizes : list
        List of subset sizes to fit
    n_cpus : int
        Number of CPUs to use for parallel processing

    Returns
    -------
    dict : Dictionary mapping subset size (as string) to line fit results
    """
    # Create tasks for parallel processing
    tasks = [(points[:n], n) for n in subset_sizes if n <= len(points)]

    # Process in parallel
    with Pool(processes=n_cpus) as pool:
        results = pool.map(fit_line_with_metadata, tasks)

    # Convert to dictionary format expected by plotting function
    line_fits = {}
    for result in results:
        n = result['n_points']
        line_fits[str(n)] = result

    return line_fits


def calculate_shared_direction_for_subset(args):
    """
    Calculate shared direction from combined NT and Bach2 points.
    Each dataset keeps its own centroid but uses the shared direction.

    Parameters
    ----------
    args : tuple
        (nt_subset, bach2_subset, n) where subsets are the points and n is subset size

    Returns
    -------
    dict : Contains shared direction info for both datasets
    """
    nt_subset, bach2_subset, n = args

    # Calculate individual centroids FIRST
    nt_centroid = np.mean(nt_subset[:, :3], axis=0)
    bach2_centroid = np.mean(bach2_subset[:, :3], axis=0)

    # Center each dataset around its OWN centroid
    nt_centered = nt_subset[:, :3] - nt_centroid
    bach2_centered = bach2_subset[:, :3] - bach2_centroid

    # NOW combine the already-centered data
    combined_centered = np.vstack([nt_centered, bach2_centered])

    # Apply PCA to find shared direction (data is already centered)
    pca = PCA(n_components=1)
    pca.fit(combined_centered)
    direction = pca.components_[0]

    # Ensure consistent orientation (positive p_self component)
    if direction[0] < 0:
        direction = -direction

    # Create parametric equations
    nt_equation = (f"p_sym = {nt_centroid[0]:.4f} + t*{direction[0]:.4f}, "
                   f"p_asym = {nt_centroid[1]:.4f} + t*{direction[1]:.4f}, "
                   f"p_dif = {nt_centroid[2]:.4f} + t*{direction[2]:.4f}")

    bach2_equation = (f"p_sym = {bach2_centroid[0]:.4f} + t*{direction[0]:.4f}, "
                      f"p_asym = {bach2_centroid[1]:.4f} + t*{direction[1]:.4f}, "
                      f"p_dif = {bach2_centroid[2]:.4f} + t*{direction[2]:.4f}")

    return {
        'n_points': n,
        'nt_shared': {
            'centroid': nt_centroid.tolist(),
            'direction': direction.tolist(),
            'parametric_equation': nt_equation
        },
        'bach2_shared': {
            'centroid': bach2_centroid.tolist(),
            'direction': direction.tolist(),
            'parametric_equation': bach2_equation
        },
        'shared_direction': direction.tolist()
    }


def calculate_shared_directions(nt_points, bach2_points, subset_sizes, n_cpus=32):
    """
    Calculate shared directions for all subset sizes in parallel.

    Parameters
    ----------
    nt_points : np.ndarray
        NT dataset lowest loss points
    bach2_points : np.ndarray
        Bach2 dataset lowest loss points
    subset_sizes : list
        List of subset sizes to process
    n_cpus : int
        Number of CPUs for parallel processing

    Returns
    -------
    dict : Dictionary mapping subset size to shared direction results
    """
    # Create tasks for parallel processing
    tasks = []
    for n in subset_sizes:
        if n <= min(len(nt_points), len(bach2_points)):
            tasks.append((nt_points[:n], bach2_points[:n], n))

    # Process in parallel
    with Pool(processes=n_cpus) as pool:
        results = pool.map(calculate_shared_direction_for_subset, tasks)

    # Convert to dictionary
    shared_directions = {}
    for result in results:
        n = result['n_points']
        shared_directions[str(n)] = result

    return shared_directions


def calculate_perpendicular_vector(nt_centroid, bach2_centroid, shared_direction):
    """
    Calculate the shortest perpendicular vector between two parallel lines.

    Parameters
    ----------
    nt_centroid : array-like
        Centroid of the NT shared line
    bach2_centroid : array-like
        Centroid of the Bach2 shared line
    shared_direction : array-like
        Shared direction vector (unit vector)

    Returns
    -------
    perpendicular_vector : np.ndarray
        The perpendicular vector from NT to Bach2 line
    perpendicular_distance : float
        The magnitude of the perpendicular vector
    """
    # Convert to numpy arrays
    nt_c = np.array(nt_centroid)
    bach2_c = np.array(bach2_centroid)
    direction = np.array(shared_direction)

    # Vector from NT to Bach2 centroid
    centroid_diff = bach2_c - nt_c

    # Project out the component parallel to the shared direction
    parallel_component = np.dot(centroid_diff, direction) * direction

    # The perpendicular vector is what remains
    perpendicular_vector = centroid_diff - parallel_component

    # Calculate the perpendicular distance
    perpendicular_distance = np.linalg.norm(perpendicular_vector)

    return perpendicular_vector, perpendicular_distance


def create_line_points(line_params, t_range=(-0.3, 0.3), n_points=100):
    """
    Create points along a parametric line for visualization.

    Parameters
    ----------
    line_params : dict
        Dictionary with 'centroid' and 'direction' for the line
    t_range : tuple
        Range of the parameter t for the parametric line
    n_points : int
        Number of points to generate along the line

    Returns
    -------
    x, y, z : arrays
        Coordinates of points along the line
    """
    centroid = np.array(line_params['centroid'])
    direction = np.array(line_params['direction'])

    # Generate parameter values
    t_values = np.linspace(t_range[0], t_range[1], n_points)

    # Generate points along the line
    points = []
    for t in t_values:
        point = centroid + t * direction
        # Only keep points that are valid (within [0,1] and sum <= 1)
        if np.all(point >= 0) and np.all(point <= 1) and np.sum(point) <= 1.0:
            points.append(point)

    if len(points) == 0:
        return np.array([]), np.array([]), np.array([])

    points = np.array(points)
    return points[:, 0], points[:, 1], points[:, 2]


def create_interactive_3d_plot(nt_points, bach2_points, nt_lines, bach2_lines, shared_directions, output_path):
    """
    Create an interactive 3D scatter plot with a slider to control the number of points shown
    and their corresponding best-fit lines (both individual and shared directions).

    Parameters
    ----------
    nt_points : np.ndarray
        NT dataset points (shape: n_points x 4)
    bach2_points : np.ndarray
        Bach2 dataset points (shape: n_points x 4)
    nt_lines : dict
        Pre-calculated line parameters for NT
    bach2_lines : dict
        Pre-calculated line parameters for Bach2
    shared_directions : dict
        Pre-calculated shared direction parameters
    output_path : str
        Path where the HTML file will be saved
    """

    # Create figure
    fig = go.Figure()

    # Generate subset sizes: 50, 60, 70, ..., 500
    subset_sizes = list(range(50, 501, 10))

    # Add persistent legend-only traces so legend never disappears when slider changes
    legend_traces = [
        # Points
        go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                     name='NT', marker=dict(size=8, color='darkblue', opacity=0.8,
                     line=dict(width=0.5, color='darkblue')), showlegend=True, visible='legendonly', legendgroup='NT'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                     name='Bach2', marker=dict(size=8, color='darkred', opacity=0.8,
                     line=dict(width=0.5, color='darkred')), showlegend=True, visible='legendonly', legendgroup='Bach2'),
        # Lines
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     name='NT Line', line=dict(color='darkblue', width=8),
                     showlegend=True, visible='legendonly', legendgroup='NT_own'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     name='Bach2 Line', line=dict(color='darkred', width=8),
                     showlegend=True, visible='legendonly', legendgroup='Bach2_own'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     name='NT Line Shared Direction', line=dict(color='cyan', width=8, dash='dash'),
                     showlegend=True, visible='legendonly', legendgroup='NT_shared'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     name='Bach2 Line Shared Direction', line=dict(color='orange', width=8, dash='dash'),
                     showlegend=True, visible='legendonly', legendgroup='Bach2_shared'),
    ]
    for lt in legend_traces:
        fig.add_trace(lt)

    legend_stub_count = len(legend_traces)

    # Create traces for each slider position
    for i, n in enumerate(subset_sizes):
        if n > len(nt_points) or n > len(bach2_points):
            continue

        # Data traces should not create duplicate legend entries; keep legend-only stubs above
        show_legend = False

        # NT trace
        nt_subset = nt_points[:n]
        # Scale point sizes inversely with loss (smaller loss = larger point)
        nt_sizes = 15 * (1.0 / (1.0 + 100 * nt_subset[:, 3]))

        # Calculate color based on loss - use loss values directly for gradient
        # Lower loss will be darker in the reversed colorscale

        fig.add_trace(
            go.Scatter3d(
                x=nt_subset[:, 0],  # p_self
                y=nt_subset[:, 1],  # p_split_one
                z=nt_subset[:, 2],  # p_split_two
                mode='markers',
                name='NT',
                marker=dict(
                    size=nt_sizes,
                    color=nt_subset[:, 3],  # Color by loss value
                    colorscale='Blues_r',  # Reversed so lower loss is darker
                    cmin=nt_subset[:, 3].min(),
                    cmax=nt_subset[:, 3].max(),
                    showscale=False,
                    opacity=0.8,
                    line=dict(width=0.5, color='darkblue')
                ),
                text=[f'NT Point {i+1}<br>' +
                      f'p_sym: {p[0]:.4f}<br>' +
                      f'p_asym: {p[1]:.4f}<br>' +
                      f'p_dif: {p[2]:.4f}<br>' +
                      f'Loss: {p[3]:.6f}'
                      for i, p in enumerate(nt_subset)],
                hovertemplate='%{text}<extra></extra>',
                visible=True,  # Show for all slider positions (will be masked by slider state)
                legendgroup='NT',
                showlegend=show_legend  # Show legend for first set
            )
        )

        # Bach2 trace
        bach2_subset = bach2_points[:n]
        # Scale point sizes inversely with loss (smaller loss = larger point)
        bach2_sizes = 15 * (1.0 / (1.0 + 100 * bach2_subset[:, 3]))

        # Calculate color based on loss - use loss values directly for gradient
        # Lower loss will be darker in the reversed colorscale

        fig.add_trace(
            go.Scatter3d(
                x=bach2_subset[:, 0],  # p_self
                y=bach2_subset[:, 1],  # p_split_one
                z=bach2_subset[:, 2],  # p_split_two
                mode='markers',
                name='Bach2',
                marker=dict(
                    size=bach2_sizes,
                    color=bach2_subset[:, 3],  # Color by loss value
                    colorscale='Reds_r',  # Reversed so lower loss is darker
                    cmin=bach2_subset[:, 3].min(),
                    cmax=bach2_subset[:, 3].max(),
                    showscale=False,
                    opacity=0.8,
                    line=dict(width=0.5, color='darkred')
                ),
                text=[f'Bach2 Point {i+1}<br>' +
                      f'p_sym: {p[0]:.4f}<br>' +
                      f'p_asym: {p[1]:.4f}<br>' +
                      f'p_dif: {p[2]:.4f}<br>' +
                      f'Loss: {p[3]:.6f}'
                      for i, p in enumerate(bach2_subset)],
                hovertemplate='%{text}<extra></extra>',
                visible=True,  # Show for all slider positions (will be masked by slider state)
                legendgroup='Bach2',
                showlegend=show_legend  # Show legend for first set
            )
        )

        # Add NT line with its own direction
        if str(n) in nt_lines:
            nt_line_params = nt_lines[str(n)]['line_params']
            x, y, z = create_line_points(nt_line_params, t_range=(-0.3, 0.3), n_points=100)

            if len(x) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name='NT Line',
                        line=dict(
                            color='darkblue',
                            width=8
                        ),
                        legendgroup='NT_own',
                        showlegend=show_legend,
                        visible=True,
                        hovertemplate=f"NT Line<br>{nt_lines[str(n)]['parametric_equation']}<extra></extra>"
                    )
                )

        # Add Bach2 line with its own direction
        if str(n) in bach2_lines:
            bach2_line_params = bach2_lines[str(n)]['line_params']
            x, y, z = create_line_points(bach2_line_params, t_range=(-0.3, 0.3), n_points=100)

            if len(x) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name='Bach2 Line',
                        line=dict(
                            color='darkred',
                            width=8
                        ),
                        legendgroup='Bach2_own',
                        showlegend=show_legend,
                        visible=True,
                        hovertemplate=f"Bach2 Line<br>{bach2_lines[str(n)]['parametric_equation']}<extra></extra>"
                    )
                )

        # Add NT line with shared direction
        if str(n) in shared_directions:
            nt_shared_params = shared_directions[str(n)]['nt_shared']
            x, y, z = create_line_points(nt_shared_params, t_range=(-0.3, 0.3), n_points=100)

            if len(x) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name='NT Line Shared Direction',
                        line=dict(
                            color='cyan',
                            width=8,
                            dash='dash'
                        ),
                        legendgroup='NT_shared',
                        showlegend=show_legend,
                        visible=True,
                        hovertemplate=f"NT Line Shared Direction<br>{nt_shared_params['parametric_equation']}<extra></extra>"
                    )
                )

        # Add Bach2 line with shared direction
        if str(n) in shared_directions:
            bach2_shared_params = shared_directions[str(n)]['bach2_shared']
            x, y, z = create_line_points(bach2_shared_params, t_range=(-0.3, 0.3), n_points=100)

            if len(x) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name='Bach2 Line Shared Direction',
                        line=dict(
                            color='orange',
                            width=8,
                            dash='dash'
                        ),
                        legendgroup='Bach2_shared',
                        showlegend=show_legend,
                        visible=True,
                        hovertemplate=f"Bach2 Line Shared Direction<br>{bach2_shared_params['parametric_equation']}<extra></extra>"
                    )
                )

    # Create slider steps with equation annotations
    steps = []
    n_valid_sizes = len([n for n in subset_sizes if n <= min(len(nt_points), len(bach2_points))])

    for i, n in enumerate(subset_sizes):
        if n > min(len(nt_points), len(bach2_points)):
            break

        # Get line parameters for equations display
        nt_c = nt_lines[str(n)]['line_params']['centroid'] if str(n) in nt_lines else [0, 0, 0]
        nt_d = nt_lines[str(n)]['line_params']['direction'] if str(n) in nt_lines else [0, 0, 0]
        bach2_c = bach2_lines[str(n)]['line_params']['centroid'] if str(n) in bach2_lines else [0, 0, 0]
        bach2_d = bach2_lines[str(n)]['line_params']['direction'] if str(n) in bach2_lines else [0, 0, 0]
        shared_d = shared_directions[str(n)]['shared_direction'] if str(n) in shared_directions else [0, 0, 0]

        # Calculate perpendicular vector between shared lines
        perp_vector_text = ""
        if str(n) in shared_directions:
            nt_shared_c = shared_directions[str(n)]['nt_shared']['centroid']
            bach2_shared_c = shared_directions[str(n)]['bach2_shared']['centroid']
            perp_vector, perp_distance = calculate_perpendicular_vector(nt_shared_c, bach2_shared_c, shared_d)
            perp_vector_text = (
                f"<br><span style='color:purple'><b>Perpendicular vector (NT→Bach2 shared):</b> "
                f"[{perp_vector[0]:.4f}, {perp_vector[1]:.4f}, {perp_vector[2]:.4f}], distance = {perp_distance:.4f}</span>"
            )

        # Create equation text with color coding
        equation_text = (
            f"<span style='color:darkblue'><b>NT Line:</b> p = [{nt_c[0]:.3f}, {nt_c[1]:.3f}, {nt_c[2]:.3f}] + t·[{nt_d[0]:.3f}, {nt_d[1]:.3f}, {nt_d[2]:.3f}]</span><br>"
            f"<span style='color:darkred'><b>Bach2 Line:</b> p = [{bach2_c[0]:.3f}, {bach2_c[1]:.3f}, {bach2_c[2]:.3f}] + t·[{bach2_d[0]:.3f}, {bach2_d[1]:.3f}, {bach2_d[2]:.3f}]</span><br>"
            f"<span style='color:cyan'><b>NT Shared:</b> p = [{nt_c[0]:.3f}, {nt_c[1]:.3f}, {nt_c[2]:.3f}] + t·[{shared_d[0]:.3f}, {shared_d[1]:.3f}, {shared_d[2]:.3f}]</span><br>"
            f"<span style='color:orange'><b>Bach2 Shared:</b> p = [{bach2_c[0]:.3f}, {bach2_c[1]:.3f}, {bach2_c[2]:.3f}] + t·[{shared_d[0]:.3f}, {shared_d[1]:.3f}, {shared_d[2]:.3f}]</span>"
            f"{perp_vector_text}<br>"
            f"<br><span style='color:gray; font-size:12px'>p = [p_sym, p_asym, p_dif] = [centroid] + t·[direction]</span>"
        )

        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {
                    "title": f"Lowest Loss Points (Showing {n} points per dataset) | Centroid distance: {np.linalg.norm(np.array(nt_lines[str(n)]['line_params']['centroid']) - np.array(bach2_lines[str(n)]['line_params']['centroid'])):.4f}",
                    "annotations": [{
                        "text": equation_text,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,  # Center horizontally
                        "y": 1.05,   # Above the plot, near title
                        "xanchor": "center",
                        "yanchor": "bottom",
                        "showarrow": False,
                        "font": {"size": 16, "family": "monospace"},
                        "align": "center",
                        "bgcolor": "rgba(255,255,255,0.95)",
                        "bordercolor": "gray",
                        "borderwidth": 1,
                        "borderpad": 4
                    }]
                    ,
                    "showlegend": True
                }
            ],
            label=str(n)
        )
        # Preserve legend-only stubs across slider updates
        for s in range(legend_stub_count):
            step["args"][0]["visible"][s] = 'legendonly'

        # Set visibility for this step (6 traces per step: NT points, Bach2 points, 4 line types)
        trace_idx = legend_stub_count + i * 6
        step["args"][0]["visible"][trace_idx] = True      # NT points
        step["args"][0]["visible"][trace_idx + 1] = True  # Bach2 points
        step["args"][0]["visible"][trace_idx + 2] = True  # NT line
        step["args"][0]["visible"][trace_idx + 3] = True  # Bach2 line
        step["args"][0]["visible"][trace_idx + 4] = True  # NT shared
        step["args"][0]["visible"][trace_idx + 5] = True  # Bach2 shared
        steps.append(step)

    # Find index for n=100 (default starting position)
    default_idx = subset_sizes.index(100) if 100 in subset_sizes else 0

    # Add slider
    sliders = [dict(
        active=default_idx,  # Start with 100 points
        currentvalue=dict(
            prefix="Number of points: ",
            visible=True,
            font=dict(size=14)
        ),
        pad=dict(t=50, b=10),
        len=0.9,
        x=0.05,
        xanchor="left",
        y=0,
        steps=steps
    )]

    # Get initial equation parameters for n=100
    initial_n = 100
    if str(initial_n) in nt_lines and str(initial_n) in bach2_lines and str(initial_n) in shared_directions:
        nt_c = nt_lines[str(initial_n)]['line_params']['centroid']
        nt_d = nt_lines[str(initial_n)]['line_params']['direction']
        bach2_c = bach2_lines[str(initial_n)]['line_params']['centroid']
        bach2_d = bach2_lines[str(initial_n)]['line_params']['direction']
        shared_d = shared_directions[str(initial_n)]['shared_direction']

        # Calculate perpendicular vector for initial display
        nt_shared_c = shared_directions[str(initial_n)]['nt_shared']['centroid']
        bach2_shared_c = shared_directions[str(initial_n)]['bach2_shared']['centroid']
        perp_vector, perp_distance = calculate_perpendicular_vector(nt_shared_c, bach2_shared_c, shared_d)
        perp_vector_text = (
            f"<br><span style='color:purple'><b>Perpendicular vector (NT→Bach2 shared):</b> "
            f"[{perp_vector[0]:.4f}, {perp_vector[1]:.4f}, {perp_vector[2]:.4f}], distance = {perp_distance:.4f}</span>"
        )

        initial_equation_text = (
            f"<span style='color:darkblue'><b>NT Line:</b> p = [{nt_c[0]:.3f}, {nt_c[1]:.3f}, {nt_c[2]:.3f}] + t·[{nt_d[0]:.3f}, {nt_d[1]:.3f}, {nt_d[2]:.3f}]</span><br>"
            f"<span style='color:darkred'><b>Bach2 Line:</b> p = [{bach2_c[0]:.3f}, {bach2_c[1]:.3f}, {bach2_c[2]:.3f}] + t·[{bach2_d[0]:.3f}, {bach2_d[1]:.3f}, {bach2_d[2]:.3f}]</span><br>"
            f"<span style='color:cyan'><b>NT Shared:</b> p = [{nt_c[0]:.3f}, {nt_c[1]:.3f}, {nt_c[2]:.3f}] + t·[{shared_d[0]:.3f}, {shared_d[1]:.3f}, {shared_d[2]:.3f}]</span><br>"
            f"<span style='color:orange'><b>Bach2 Shared:</b> p = [{bach2_c[0]:.3f}, {bach2_c[1]:.3f}, {bach2_c[2]:.3f}] + t·[{shared_d[0]:.3f}, {shared_d[1]:.3f}, {shared_d[2]:.3f}]</span>"
            f"{perp_vector_text}<br>"
            f"<br><span style='color:gray; font-size:12px'>p = [p_sym, p_asym, p_dif] = [centroid] + t·[direction]</span>"
        )

        # Calculate centroid distance
        centroid_distance = np.linalg.norm(np.array(nt_c) - np.array(bach2_c))
        initial_title = f"Lowest Loss Points (Showing 100 points per dataset) | Centroid distance: {centroid_distance:.4f}"
    else:
        initial_equation_text = "Line equations will appear here"
        initial_title = "Lowest Loss Points (Showing 100 points per dataset)"

    # Update layout
    fig.update_layout(
        title=initial_title,
        showlegend=True,
        scene=dict(
            xaxis=dict(
                title="p_sym",
                range=[0, 0.6],
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(230, 230, 230, 0.1)"
            ),
            yaxis=dict(
                title="p_asym",
                range=[0, 0.4],
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(230, 230, 230, 0.1)"
            ),
            zaxis=dict(
                title="p_dif",
                range=[0, 0.4],
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(230, 230, 230, 0.1)"
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        sliders=sliders,
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=100, t=170),  # Moderate top margin for equations near title
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        font=dict(family="Arial, sans-serif", size=12),
        annotations=[{
            "text": initial_equation_text,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,  # Center horizontally
            "y": 1.05,   # Above the plot, near title
            "xanchor": "center",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 16, "family": "monospace"},
            "align": "center",
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "gray",
            "borderwidth": 1,
            "borderpad": 4
        }]
    )

    # Set the initial visibility to match the default slider index
    # Hide all data traces first
    for idx in range(legend_stub_count, len(fig.data)):
        fig.data[idx].visible = False

    # Keep legend-only stubs as legendonly
    for s in range(legend_stub_count):
        fig.data[s].visible = 'legendonly'

    # Turn on the traces for the default position
    default_idx = subset_sizes.index(100) if 100 in subset_sizes else 0
    start = legend_stub_count + default_idx * 6
    for j in range(6):
        if start + j < len(fig.data):
            fig.data[start + j].visible = True

    # Ensure destination directory exists and save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive plot saved to: {output_path}")

    return fig


def print_summary_statistics(nt_points, bach2_points):
    """
    Print summary statistics about the lowest loss points.
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # NT statistics
    print("\nNT Dataset (100 lowest loss points):")
    print(f"  Loss range: {nt_points[0, 3]:.6f} - {nt_points[-1, 3]:.6f}")
    print(f"  Mean loss: {np.mean(nt_points[:, 3]):.6f}")
    print(f"  Median loss: {np.median(nt_points[:, 3]):.6f}")
    print(f"  Best point: p_sym={nt_points[0, 0]:.4f}, "
          f"p_asym={nt_points[0, 1]:.4f}, "
          f"p_dif={nt_points[0, 2]:.4f}")

    # Bach2 statistics
    print("\nBach2 Dataset (100 lowest loss points):")
    print(f"  Loss range: {bach2_points[0, 3]:.6f} - {bach2_points[-1, 3]:.6f}")
    print(f"  Mean loss: {np.mean(bach2_points[:, 3]):.6f}")
    print(f"  Median loss: {np.median(bach2_points[:, 3]):.6f}")
    print(f"  Best point: p_sym={bach2_points[0, 0]:.4f}, "
          f"p_asym={bach2_points[0, 1]:.4f}, "
          f"p_dif={bach2_points[0, 2]:.4f}")

    print("\nParameter ranges for lowest loss regions:")
    print("\nNT:")
    print(f"  p_sym: [{np.min(nt_points[:, 0]):.4f}, {np.max(nt_points[:, 0]):.4f}]")
    print(f"  p_asym: [{np.min(nt_points[:, 1]):.4f}, {np.max(nt_points[:, 1]):.4f}]")
    print(f"  p_dif: [{np.min(nt_points[:, 2]):.4f}, {np.max(nt_points[:, 2]):.4f}]")

    print("\nBach2:")
    print(f"  p_sym: [{np.min(bach2_points[:, 0]):.4f}, {np.max(bach2_points[:, 0]):.4f}]")
    print(f"  p_asym: [{np.min(bach2_points[:, 1]):.4f}, {np.max(bach2_points[:, 1]):.4f}]")
    print(f"  p_dif: [{np.min(bach2_points[:, 2]):.4f}, {np.max(bach2_points[:, 2]):.4f}]")

    print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create interactive 3D plot of lowest loss points for NT and Bach2 with dynamically calculated best-fit lines"
    )
    parser.add_argument(
        "--nt-data",
        type=str,
        default="./clonal_analysis/intermediate_files/gene10/NT/grid_search_loss.npz",
        help="Path to NT loss landscape data file"
    )
    parser.add_argument(
        "--bach2-data",
        type=str,
        default="./clonal_analysis/intermediate_files/gene10/Bach2/grid_search_loss.npz",
        help="Path to Bach2 loss landscape data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./clonal_analysis/intermediate_files/gene10/lowest_loss_points_3d.html",
        help="Output path for HTML file"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=1000,
        help="Maximum number of lowest loss points to extract per dataset (default: 1000)"
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=32,
        help="Number of CPUs to use for parallel line fitting (default: 32)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print("Loading loss landscape data...")
    print(f"  NT data: {args.nt_data}")
    print(f"  Bach2 data: {args.bach2_data}")

    # Extract lowest loss points
    print(f"\nExtracting {args.n_points} lowest loss points from each dataset...")
    nt_points = load_and_extract_lowest_points(args.nt_data, args.n_points)
    bach2_points = load_and_extract_lowest_points(args.bach2_data, args.n_points)

    print(f"  Found {len(nt_points)} NT points")
    print(f"  Found {len(bach2_points)} Bach2 points")

    # Calculate line fits dynamically using parallel processing
    print(f"\nCalculating line fits dynamically using {args.n_cpus} CPUs...")
    subset_sizes = list(range(50, min(501, min(len(nt_points), len(bach2_points)) + 1), 10))
    print(f"  Fitting lines for {len(subset_sizes)} subset sizes (50 to {subset_sizes[-1] if subset_sizes else 0} points)")

    print("  Calculating NT line fits...")
    nt_lines = calculate_line_fits_parallel(nt_points, subset_sizes, args.n_cpus)
    print(f"    Completed {len(nt_lines)} NT line fits")

    print("  Calculating Bach2 line fits...")
    bach2_lines = calculate_line_fits_parallel(bach2_points, subset_sizes, args.n_cpus)
    print(f"    Completed {len(bach2_lines)} Bach2 line fits")

    print("  Calculating shared directions from combined data...")
    shared_directions = calculate_shared_directions(nt_points, bach2_points, subset_sizes, args.n_cpus)
    print(f"    Completed {len(shared_directions)} shared direction calculations")

    # Debug: Show how shared direction differs from individual directions
    if '100' in nt_lines and '100' in bach2_lines and '100' in shared_directions:
        print("\n  Direction comparison for n=100:")
        print(f"    NT direction:     {[f'{x:.3f}' for x in nt_lines['100']['line_params']['direction']]}")
        print(f"    Bach2 direction:  {[f'{x:.3f}' for x in bach2_lines['100']['line_params']['direction']]}")
        print(f"    Shared direction: {[f'{x:.3f}' for x in shared_directions['100']['shared_direction']]}")

    # Print summary statistics
    print_summary_statistics(nt_points[:100], bach2_points[:100])

    # Create interactive plot
    print("\nCreating interactive 3D visualization with best-fit lines...")
    create_interactive_3d_plot(nt_points, bach2_points, nt_lines, bach2_lines, shared_directions, args.output)

    print(f"\n✓ Successfully created interactive plot!")
    print(f"  Open {args.output} in a web browser to interact with the visualization.")
    print("\nInteractive features:")
    print("  - Drag to rotate the 3D plot")
    print("  - Scroll to zoom in/out")
    print("  - Use the slider to control number of points shown (50-500, increments of 10)")
    print("  - Best-fit lines update based on the number of points shown")
    print("  - Line equations displayed near the title, color-coded to match lines")
    print("  - Hover over points to see parameter values and loss")
    print("  - Hover over lines to see their parametric equations")
    print("  - Lower loss points have higher opacity (more visible)")
    print("  - Click legend items to toggle on/off:")
    print("    • NT and Bach2 point clouds")
    print("    • NT Line - dark blue solid (own PCA direction)")
    print("    • Bach2 Line - dark red solid (own PCA direction)")
    print("    • NT Line Shared Direction - cyan dashed (shared direction from combined data)")
    print("    • Bach2 Line Shared Direction - orange dashed (shared direction from combined data)")


if __name__ == "__main__":
    main()
