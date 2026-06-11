#!/usr/bin/env python
"""
Create loss landscape visualizations.
Supports two modes:
- 3d: Interactive 3D surface plots with sliders (generates HTML files)
- heatmap: 4x4 grid of 2D heatmaps (generates PNG/PDF files)
"""

import os
import numpy as np
# Plotly is optional; heatmap generation works without it.
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - graceful degradation when plotly missing.
    go = None
    make_subplots = None
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import argparse

def create_surface_plot_pself_slider(p_self_values, p_split_one_values, p_split_two_values, loss_grid, max_loss=0.2):
    """
    Create surface plot with p_sym as slider.
    X-axis: p_asym, Y-axis: p_dif, Z-axis: loss

    Parameters
    ----------
    max_loss : float, default 0.2
        Maximum loss value for z-axis range and colorscale
    """

    if go is None:
        raise ImportError("plotly is required for 3D surface plots but is not installed.")

    fig = go.Figure()

    # Create meshgrid for X and Y
    X, Y = np.meshgrid(p_split_one_values, p_split_two_values)

    # Add a surface for each p_self value
    for i, p_self in enumerate(p_self_values):
        # Extract 2D slice where p_self is fixed
        loss_slice = loss_grid[i, :, :].T  # Transpose for correct orientation

        # Apply constraint mask
        constraint_mask = (p_self + X + Y) > 1
        loss_slice_masked = loss_slice.copy()
        loss_slice_masked[constraint_mask] = np.nan

        # Skip if all values are NaN
        if np.all(np.isnan(loss_slice_masked)):
            loss_slice_masked = np.full_like(loss_slice, np.nan)

        fig.add_trace(
            go.Surface(
                x=p_split_one_values,
                y=p_split_two_values,
                z=loss_slice_masked,
                visible=(i == 0),  # Only first surface is visible initially
                colorscale='rainbow',
                showscale=True,
                cmin=0,
                cmax=max_loss,
                name=f'p_sym={p_self:.3f}',
                hovertemplate='p_asym: %{x:.3f}<br>' +
                              'p_dif: %{y:.3f}<br>' +
                              'Loss: %{z:.6f}<extra></extra>'
            )
        )

    # Create slider steps
    steps = []
    for i, p_self in enumerate(p_self_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Loss Landscape (p_sym = {p_self:.3f})"}],
            label=f"{p_self:.3f}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    # Add slider
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="p_sym = ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        y=0,
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=f"Loss Landscape (p_sym = {p_self_values[0]:.3f})",
        scene=dict(
            xaxis_title="p_asym",
            yaxis_title="p_dif",
            zaxis_title="Loss",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            xaxis=dict(range=[0, 0.2]),
            yaxis=dict(range=[0, 0.2]),
            zaxis=dict(range=[0, max_loss]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return fig

def create_surface_plot_p1_slider(p_self_values, p_split_one_values, p_split_two_values, loss_grid, max_loss=0.2):
    """
    Create surface plot with p_asym as slider.
    X-axis: p_sym, Y-axis: p_dif, Z-axis: loss

    Parameters
    ----------
    max_loss : float, default 0.2
        Maximum loss value for z-axis range and colorscale
    """

    if go is None:
        raise ImportError("plotly is required for 3D surface plots but is not installed.")

    fig = go.Figure()

    # Create meshgrid for X and Y
    X, Y = np.meshgrid(p_self_values, p_split_two_values)

    # Add a surface for each p_split_one value
    for j, p_split_one in enumerate(p_split_one_values):
        # Extract 2D slice where p_split_one is fixed
        loss_slice = loss_grid[:, j, :].T  # Transpose for correct orientation

        # Apply constraint mask
        constraint_mask = (X + p_split_one + Y) > 1
        loss_slice_masked = loss_slice.copy()
        loss_slice_masked[constraint_mask] = np.nan

        # Skip if all values are NaN
        if np.all(np.isnan(loss_slice_masked)):
            loss_slice_masked = np.full_like(loss_slice, np.nan)

        fig.add_trace(
            go.Surface(
                x=p_self_values,
                y=p_split_two_values,
                z=loss_slice_masked,
                visible=(j == 0),  # Only first surface is visible initially
                colorscale='rainbow',
                showscale=True,
                cmin=0,
                cmax=max_loss,
                name=f'p_asym={p_split_one:.3f}',
                hovertemplate='p_sym: %{x:.3f}<br>' +
                              'p_dif: %{y:.3f}<br>' +
                              'Loss: %{z:.6f}<extra></extra>'
            )
        )

    # Create slider steps
    steps = []
    for j, p_split_one in enumerate(p_split_one_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Loss Landscape (p_asym = {p_split_one:.3f})"}],
            label=f"{p_split_one:.3f}"
        )
        step["args"][0]["visible"][j] = True
        steps.append(step)

    # Add slider
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="p_asym = ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        y=0,
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=f"Loss Landscape (p_asym = {p_split_one_values[0]:.3f})",
        scene=dict(
            xaxis_title="p_sym",
            yaxis_title="p_dif",
            zaxis_title="Loss",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            xaxis=dict(range=[0.2, 0.6]),
            yaxis=dict(range=[0, 0.2]),
            zaxis=dict(range=[0, max_loss]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return fig

def create_surface_plot_p2_slider(p_self_values, p_split_one_values, p_split_two_values, loss_grid, max_loss=0.2):
    """
    Create surface plot with p_dif as slider.
    X-axis: p_sym, Y-axis: p_asym, Z-axis: loss

    Parameters
    ----------
    max_loss : float, default 0.2
        Maximum loss value for z-axis range and colorscale
    """

    if go is None:
        raise ImportError("plotly is required for 3D surface plots but is not installed.")

    fig = go.Figure()

    # Create meshgrid for X and Y
    X, Y = np.meshgrid(p_self_values, p_split_one_values)

    # Add a surface for each p_split_two value
    for k, p_split_two in enumerate(p_split_two_values):
        # Extract 2D slice where p_split_two is fixed
        loss_slice = loss_grid[:, :, k].T  # Transpose for correct orientation

        # Apply constraint mask
        constraint_mask = (X + Y + p_split_two) > 1
        loss_slice_masked = loss_slice.copy()
        loss_slice_masked[constraint_mask] = np.nan

        # Skip if all values are NaN
        if np.all(np.isnan(loss_slice_masked)):
            loss_slice_masked = np.full_like(loss_slice, np.nan)

        fig.add_trace(
            go.Surface(
                x=p_self_values,
                y=p_split_one_values,
                z=loss_slice_masked,
                visible=(k == 0),  # Only first surface is visible initially
                colorscale='rainbow',
                showscale=True,
                cmin=0,
                cmax=max_loss,
                name=f'p_dif={p_split_two:.3f}',
                hovertemplate='p_sym: %{x:.3f}<br>' +
                              'p_asym: %{y:.3f}<br>' +
                              'Loss: %{z:.6f}<extra></extra>'
            )
        )

    # Create slider steps
    steps = []
    for k, p_split_two in enumerate(p_split_two_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Loss Landscape (p_dif = {p_split_two:.3f})"}],
            label=f"{p_split_two:.3f}"
        )
        step["args"][0]["visible"][k] = True
        steps.append(step)

    # Add slider
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="p_dif = ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        y=0,
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=f"Loss Landscape (p_dif = {p_split_two_values[0]:.3f})",
        scene=dict(
            xaxis_title="p_sym",
            yaxis_title="p_asym",
            zaxis_title="Loss",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            xaxis=dict(range=[0.2, 0.6]),
            yaxis=dict(range=[0, 0.2]),
            zaxis=dict(range=[0, max_loss]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        width=1000,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return fig

def determine_parameter_range(values, loss_grid, axis_idx, loss_threshold=0.003):
    """Return a tight axis range that emphasises low-loss regions."""
    finite_mask = np.isfinite(loss_grid)
    candidate_mask = finite_mask & (loss_grid <= loss_threshold)
    if not np.any(candidate_mask):
        candidate_mask = finite_mask

    axis_indices = np.where(candidate_mask)[axis_idx]
    if axis_indices.size == 0:
        return float(values[0]), float(values[-1])

    lower = float(values[axis_indices].min())
    upper = float(values[axis_indices].max())

    span = upper - lower
    step = float(values[1] - values[0]) if len(values) > 1 else 0.0
    margin = max(span * 0.1, step)

    lower = max(float(values[0]), lower - margin)
    upper = min(float(values[-1]), upper + margin)
    return lower, upper


def create_custom_colormap():
    """Create a rainbow colormap that fades to white at the highest loss values."""
    base_cmap = plt.cm.get_cmap('rainbow', 256)
    colors = base_cmap(np.linspace(0, 1, 256))

    # Fade the top of the rainbow smoothly into white so the maximum stands out clearly.
    white = np.array([1.0, 1.0, 1.0, 1.0])
    blend_steps = 12
    for i in range(1, blend_steps + 1):
        alpha = i / (blend_steps + 1)
        colors[-i] = colors[-i] * (1 - alpha) + white * alpha
    colors[-1] = white

    cmap = LinearSegmentedColormap.from_list('rainbow_to_white', colors, N=256)
    cmap.set_over('white')
    cmap.set_bad('white')
    return cmap

def extract_slice(loss_grid, param_values, fixed_param_idx, fixed_value_idx):
    """
    Extract a 2D slice from the 3D loss grid.

    Parameters:
    -----------
    loss_grid : np.ndarray
        3D array of loss values [p_self, p_split_one, p_split_two]
    param_values : tuple
        (p_self_values, p_split_one_values, p_split_two_values)
    fixed_param_idx : int
        Index of the parameter to fix (0=p_self, 1=p_split_one, 2=p_split_two)
    fixed_value_idx : int
        Index of the fixed value in the parameter array

    Returns:
    --------
    slice_2d : np.ndarray
        2D array with the loss values for the slice
    """
    if fixed_param_idx == 0:  # Fix p_self
        # Return p_split_one (rows) vs p_split_two (cols)
        return loss_grid[fixed_value_idx, :, :]
    elif fixed_param_idx == 1:  # Fix p_split_one
        # Return p_self (rows) vs p_split_two (cols)
        return loss_grid[:, fixed_value_idx, :]
    else:  # Fix p_split_two
        # Return p_self (rows) vs p_split_one (cols)
        return loss_grid[:, :, fixed_value_idx]

def create_heatmap_grid(
    p_self_values,
    p_split_one_values,
    p_split_two_values,
    loss_grid,
    output_dir=None,
    data_file=None,
    max_loss=None,
    plane_coeffs=None,
    fit_points=None,
    plane_style=None,
    point_style=None,
    plane_loss_threshold=None,
    output_filename="loss_landscape_heatmap.png",
):
    """
    Create a 4x4 grid of loss landscape heatmaps.
    Each heatmap fixes p_sym and shows p_asym vs p_dif.
    Only uses p_sym values where reasonable loss (< 0.001) exists.
    Color scheme: rainbow gradient with white at loss ≥ 0.01.

    Parameters
    ----------
    plane_coeffs : sequence of float, optional
        Coefficients (a, b, c) of the plane a·p_sym + b·p_asym + c·p_dif = 1.
        When provided, the intersection of the plane with each slice is drawn.
    fit_points : array-like with shape (n, 3), optional
        Collection of (p_sym, p_asym, p_dif) combinations used when
        fitting the plane. Points are overlaid on matching heatmap panels.
    plane_style : dict, optional
        Matplotlib styling kwargs for the plane intersection line.
    point_style : dict, optional
        Matplotlib styling kwargs for the fitting sample scatter markers.
    plane_loss_threshold : float, optional
        Restrict the plane overlay to regions where the nearest grid point has
        loss ≤ this value. Useful for trimming invalid extremes.
    output_filename : str, optional
        File name (within output_dir) for the generated PNG.
    """
    # Create custom colormap
    cmap = create_custom_colormap()

    if fit_points is not None:
        fit_points = np.asarray(fit_points, dtype=float)
        if fit_points.ndim != 2 or fit_points.shape[1] != 3:
            raise ValueError("fit_points must be an array of shape (n, 3)")

    if plane_coeffs is not None and len(plane_coeffs) != 3:
        raise ValueError("plane_coeffs must contain exactly three values (a, b, c)")

    if plane_style is None:
        plane_style = {"color": "black", "linewidth": 2.0, "linestyle": "--", "alpha": 0.9}
    if point_style is None:
        point_style = {
            "facecolors": "white",
            "edgecolors": "black",
            "s": 20,
            "linewidths": 0.6,
            "alpha": 0.9,
        }

    plane_label_drawn = False
    primary_plane = False
    point_label_drawn = False
    legend_handles = []
    legend_labels = []

    # Find p_self values with reasonable loss (at least one point < 0.001)
    valid_p_self_indices = []
    for i in range(len(p_self_values)):
        loss_slice = loss_grid[i, :, :]
        if np.any(loss_slice[~np.isnan(loss_slice)] < 0.001):
            valid_p_self_indices.append(i)

    # Select up to 16 evenly spaced indices from valid ones
    n_valid = len(valid_p_self_indices)
    threshold_used = n_valid > 0
    if n_valid == 0:
        print("Warning: No p_sym values have loss < 0.001. Selecting slices with lowest available loss.")
        slice_mins = []
        for i in range(loss_grid.shape[0]):
            slice_vals = loss_grid[i, :, :]
            finite_vals = slice_vals[~np.isnan(slice_vals)]
            slice_mins.append(np.min(finite_vals) if finite_vals.size else np.nan)
        slice_mins = np.array(slice_mins)
        finite_mask = ~np.isnan(slice_mins)
        if not np.any(finite_mask):
            raise ValueError("All loss slices are NaN; cannot generate heatmap grid.")
        sorted_indices = np.argsort(slice_mins[finite_mask])
        available_indices = np.arange(len(slice_mins))[finite_mask]
        top_indices = available_indices[sorted_indices]
        selected_indices = list(top_indices[:min(16, len(top_indices))])
    else:
        target_count = min(16, n_valid)
        linspace_idx = np.linspace(0, n_valid - 1, target_count)
        selected_indices = []
        for idx in linspace_idx:
            candidate = valid_p_self_indices[int(round(idx))]
            if candidate not in selected_indices:
                selected_indices.append(candidate)
        if len(selected_indices) < target_count:
            for candidate in valid_p_self_indices:
                if candidate not in selected_indices:
                    selected_indices.append(candidate)
                if len(selected_indices) == target_count:
                    break

    selected_indices = sorted(selected_indices)

    # Pad with repeats if fewer than 16 slices are available
    if len(selected_indices) < 16:
        print(f"Warning: Only {len(selected_indices)} unique p_sym slices available. Repeating final slice to fill grid.")
        while len(selected_indices) < 16:
            selected_indices.append(selected_indices[-1])

    # Set color scale limits
    white_cutoff = 0.01
    if max_loss is not None and max_loss < white_cutoff:
        print("Warning: Received max_loss < 0.01; clamping to 0.01 to preserve white threshold.")

    vmin = 0
    vmax = white_cutoff

    # Determine axis limits that focus on the most informative region
    p_split_one_min, p_split_one_max = determine_parameter_range(
        p_split_one_values, loss_grid, axis_idx=1, loss_threshold=0.003
    )
    p_split_two_min, p_split_two_max = determine_parameter_range(
        p_split_two_values, loss_grid, axis_idx=2, loss_threshold=0.003
    )

    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(22, 22))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.32, wspace=0.32)

    p_self_step = float(p_self_values[1] - p_self_values[0]) if len(p_self_values) > 1 else 0.0
    p_self_tol = max(p_self_step * 0.51, 1e-6)

    # Create each subplot
    for subplot_idx, p_self_idx in enumerate(selected_indices):
        row = subplot_idx // 4
        col = subplot_idx % 4

        ax = fig.add_subplot(gs[row, col])

        # Extract 2D slice for fixed p_self (shows p_split_one vs p_split_two)
        slice_2d = loss_grid[p_self_idx, :, :]
        display_slice = np.clip(slice_2d, a_min=0, a_max=white_cutoff)

        # Create heatmap
        # Note: imshow expects (rows, cols) which corresponds to (y, x)
        im = ax.imshow(display_slice,
                      cmap=cmap,
                      aspect='auto',
                      origin='lower',
                      extent=[p_split_two_values[0], p_split_two_values[-1],
                              p_split_one_values[0], p_split_one_values[-1]],
                      vmin=vmin, vmax=vmax,
                      interpolation='nearest')

        # Add title with fixed p_self value
        p_self_value = p_self_values[p_self_idx]
        ax.set_title(f'p_sym = {p_self_value:.3f}', fontsize=16)

        # Add axis labels
        ax.set_xlabel('p_dif', fontsize=14)
        ax.set_ylabel('p_asym', fontsize=14)

        # Add tick labels
        ax.tick_params(labelsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)

        # Focus on the most informative range
        ax.set_xlim(p_split_two_min, p_split_two_max)
        ax.set_ylim(p_split_one_min, p_split_one_max)

        # Overlay points used for plane fitting, if provided
        if fit_points is not None:
            target_p_self = float(p_self_values[p_self_idx])
            mask = np.abs(fit_points[:, 0] - target_p_self) <= p_self_tol
            slice_points = fit_points[mask]
            if slice_points.size:
                label = "Fitting samples" if not point_label_drawn else "_nolegend_"
                scatter = ax.scatter(
                    slice_points[:, 2],  # x-axis: p_split_two
                    slice_points[:, 1],  # y-axis: p_split_one
                    label=label,
                    **point_style,
                )
                if label != "_nolegend_":
                    legend_handles.append(scatter)
                    legend_labels.append("Fitting samples")
                    point_label_drawn = True

        # Overlay plane intersection, if provided
        if plane_coeffs is not None:
            a, b, c = map(float, plane_coeffs)
            target_p_self = float(p_self_values[p_self_idx])
            label = "Plane intersection" if not plane_label_drawn else "_nolegend_"

            # Handle planes that can be expressed as p_split_two = f(p_split_one)
            if abs(c) >= 1e-9:
                one_grid = np.linspace(p_split_one_min, p_split_one_max, 400)
                two_vals = (1.0 - (a * target_p_self) - (b * one_grid)) / c
                valid_mask = np.isfinite(two_vals)
                valid_mask &= (two_vals >= p_split_two_min) & (two_vals <= p_split_two_max)
                valid_mask &= (target_p_self + one_grid + two_vals) <= 1.0 + 1e-9
                if plane_loss_threshold is not None:
                    idx_one = np.abs(p_split_one_values[:, None] - one_grid[None, :]).argmin(axis=0)
                    idx_two = np.abs(p_split_two_values[:, None] - two_vals[None, :]).argmin(axis=0)
                    losses = slice_2d[idx_one, idx_two]
                    valid_mask &= np.isfinite(losses)
                    valid_mask &= losses <= plane_loss_threshold
                if np.any(valid_mask):
                    line, = ax.plot(
                        two_vals[valid_mask],
                        one_grid[valid_mask],
                        label=label,
                        **plane_style,
                    )
                    if label != "_nolegend_":
                        legend_handles.append(line)
                        legend_labels.append("Plane intersection")
                        plane_label_drawn = True
                    primary_plane = True
            elif abs(b) >= 1e-9:
                two_grid = np.linspace(p_split_two_min, p_split_two_max, 400)
                one_vals = (1.0 - (a * target_p_self) - (c * two_grid)) / b
                valid_mask = np.isfinite(one_vals)
                valid_mask &= (one_vals >= p_split_one_min) & (one_vals <= p_split_one_max)
                valid_mask &= (target_p_self + one_vals + two_grid) <= 1.0 + 1e-9
                if plane_loss_threshold is not None:
                    idx_two = np.abs(p_split_two_values[:, None] - two_grid[None, :]).argmin(axis=0)
                    idx_one = np.abs(p_split_one_values[:, None] - one_vals[None, :]).argmin(axis=0)
                    losses = slice_2d[idx_one, idx_two]
                    valid_mask &= np.isfinite(losses)
                    valid_mask &= losses <= plane_loss_threshold
                if np.any(valid_mask):
                    line, = ax.plot(
                        two_grid[valid_mask],
                        one_vals[valid_mask],
                        label=label,
                        **plane_style,
                    )
                    if label != "_nolegend_":
                        legend_handles.append(line)
                        legend_labels.append("Plane intersection")
                        plane_label_drawn = True
                    primary_plane = True
            else:
                # Degenerate scenario where both b and c are ~0; nothing to draw.
                pass

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax, extend='max')
    cbar.set_label('Loss (rainbow low→high, white ≥ 0.01)', fontsize=12, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Determine dataset name from file path
    dataset_name = "NT"
    if data_file:
        if "bach2" in data_file.lower():
            dataset_name = "Bach2"
        elif "nt" in data_file.lower():
            dataset_name = "NT"

    # Build descriptive title
    title_lines = [f"{dataset_name} Loss Landscape Heatmaps"]
    title_suffix = "(Rainbow low→high, white ≥ 0.01 loss)"
    if plane_coeffs is not None and primary_plane:
        a, b, c = map(float, plane_coeffs)
        title_lines.append(f"Plane: {a:.2f}·p_sym + {b:.2f}·p_asym + {c:.2f}·p_dif = 1")
    title_lines.append(title_suffix)

    fig.suptitle("\n".join(title_lines), fontsize=20, y=0.97)

    # Save figure
    if output_dir is None:
        output_dir = "./clonal_analysis/intermediate_files"

    if not output_filename:
        output_filename = "loss_landscape_heatmap.png"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, output_filename)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=len(legend_handles),
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.08 + 0.04)
    else:
        fig.subplots_adjust(bottom=0.06)

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap grid to: {png_path}")

    plt.close()

    # Print summary
    print(f"\nHeatmap grid summary:")
    print(f"  - 4x4 grid covering {len(set(selected_indices))} unique p_sym values")
    if threshold_used:
        print("  - Only includes p_sym slices with min loss < 0.001")
    else:
        print("  - No p_sym slices met loss < 0.001; using lowest-loss slices instead")
    print(f"  - Color scale: Rainbow gradient, white at loss ≥ 0.01")
    print(f"  - Each heatmap shows p_asym vs p_dif at fixed p_sym")
    print(f"  - Axis focus: p_asym {p_split_one_min:.3f}–{p_split_one_max:.3f}, "
          f"p_dif {p_split_two_min:.3f}–{p_split_two_max:.3f}")
    if plane_coeffs is not None:
        a, b, c = map(float, plane_coeffs)
        print(f"  - Plane overlay: {a:.2f}·p_sym + {b:.2f}·p_asym + {c:.2f}·p_dif = 1")
    if fit_points is not None:
        print(f"  - Overlay includes {len(fit_points)} fitting samples across all slices")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Create loss landscape visualizations (3D or heatmap)")
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=["3d", "heatmap"],
        help="Visualization mode: '3d' for interactive plots or 'heatmap' for 4x4 grid (default: 3d)"
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=0.2,
        help="Maximum loss value for z-axis range and colorscale (default: 0.2)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="./clonal_analysis/intermediate_files/loss_landscape.npz",
        help="Path to loss landscape data file (default: NT data)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="loss_landscape",
        help="Prefix for output files (default: 'loss_landscape')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for files (overrides output-prefix)"
    )
    return parser.parse_args()

def main():
    """
    Load computed loss landscape and create visualizations based on mode.
    """
    args = parse_args()

    # Load data
    print(f"Loading loss landscape data from {args.data_file}...")
    print(f"Visualization mode: {args.mode}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Loss landscape data not found at {args.data_file}. "
                              "Please run the corresponding compute script first.")

    data = np.load(args.data_file)
    p_self_values = data['p_self_values']
    p_split_one_values = data['p_split_one_values']
    p_split_two_values = data['p_split_two_values']
    loss_grid = data['loss_grid']
    n_valid = data['n_valid']

    print(f"Loaded loss grid with shape: {loss_grid.shape}")
    print(f"Number of valid parameter combinations: {n_valid}")

    if args.mode == "heatmap":
        # Create 4x4 heatmap grid
        print("\nCreating 4x4 heatmap grid...")
        print(f"Using max loss value: {args.max_loss}")
        create_heatmap_grid(p_self_values, p_split_one_values, p_split_two_values,
                           loss_grid, output_dir=args.output_dir, data_file=args.data_file,
                           max_loss=args.max_loss)
        print("\n✓ Heatmap grid created successfully!")

    else:  # 3d mode
        if go is None:
            raise ImportError(
                "Plotly is required for 3D visualisations. Install plotly or rerun with --mode heatmap."
            )
        print(f"Using max loss value: {args.max_loss}")

        # Determine output directory and file paths
        if args.output_dir:
            # Use the provided output directory
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_path_a = os.path.join(output_dir, "pself_slider.html")
            output_path_b = os.path.join(output_dir, "p1_slider.html")
            output_path_c = os.path.join(output_dir, "p2_slider.html")
        else:
            # Use the default with output_prefix
            output_dir = "./clonal_analysis/intermediate_files"
            output_path_a = os.path.join(output_dir, f"{args.output_prefix}_pself_slider.html")
            output_path_b = os.path.join(output_dir, f"{args.output_prefix}_p1_slider.html")
            output_path_c = os.path.join(output_dir, f"{args.output_prefix}_p2_slider.html")

        # Create Plot A: p_self slider
        print("\nCreating Plot A: p_sym slider...")
        fig_a = create_surface_plot_pself_slider(
            p_self_values, p_split_one_values, p_split_two_values, loss_grid, args.max_loss
        )
        fig_a.write_html(output_path_a)
        print(f"Saved to {output_path_a}")

        # Create Plot B: p_split_one slider
        print("\nCreating Plot B: p_asym slider...")
        fig_b = create_surface_plot_p1_slider(
            p_self_values, p_split_one_values, p_split_two_values, loss_grid, args.max_loss
        )
        fig_b.write_html(output_path_b)
        print(f"Saved to {output_path_b}")

        # Create Plot C: p_split_two slider
        print("\nCreating Plot C: p_dif slider...")
        fig_c = create_surface_plot_p2_slider(
            p_self_values, p_split_one_values, p_split_two_values, loss_grid, args.max_loss
        )
        fig_c.write_html(output_path_c)
        print(f"Saved to {output_path_c}")

        print("\n✓ All 3D visualizations created successfully!")
        print("\nTo view the interactive plots, open the HTML files in a web browser:")
        print(f"  1. {output_path_a}")
        print(f"  2. {output_path_b}")
        print(f"  3. {output_path_c}")

if __name__ == "__main__":
    main()
