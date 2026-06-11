#!/usr/bin/env python
"""
Create an interactive 3D scatter plot for an arbitrary set of datasets, mirroring the
plot_lowest_loss_points_3d analysis but allowing more than two perturbations.
Each dataset contributes its lowest-loss points, per-dataset PCA line, and a shared
direction line calculated from the combined centered clouds.
"""

import argparse
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from plot_lowest_loss_points_3d import (
    load_and_extract_lowest_points,
    calculate_line_fits_parallel,
    create_line_points
)


STYLE_PRESETS = OrderedDict({
    "NT": {
        "marker_color": "darkblue",
        "colorscale": "Blues_r",
        "line_color": "darkblue",
        "shared_color": "cyan"
    },
    "Bach2": {
        "marker_color": "darkred",
        "colorscale": "Reds_r",
        "line_color": "darkred",
        "shared_color": "orange"
    },
    "NT_trim28_setdb1": {
        "marker_color": "midnightblue",
        "colorscale": "Blues",
        "line_color": "midnightblue",
        "shared_color": "royalblue"
    },
    "Trim28": {
        "marker_color": "darkgreen",
        "colorscale": "Greens_r",
        "line_color": "darkgreen",
        "shared_color": "limegreen"
    },
    "Setdb1": {
        "marker_color": "purple",
        "colorscale": "Purples_r",
        "line_color": "purple",
        "shared_color": "magenta"
    }
})

FALLBACK_STYLES = [
    {
        "marker_color": "#636EFA",
        "colorscale": "Viridis",
        "line_color": "#636EFA",
        "shared_color": "#AB63FA"
    },
    {
        "marker_color": "#EF553B",
        "colorscale": "Inferno",
        "line_color": "#EF553B",
        "shared_color": "#FFA15A"
    },
    {
        "marker_color": "#00CC96",
        "colorscale": "Greens",
        "line_color": "#00CC96",
        "shared_color": "#19D3F3"
    },
    {
        "marker_color": "#FF6692",
        "colorscale": "Reds",
        "line_color": "#FF6692",
        "shared_color": "#B6E880"
    }
]


def parse_dataset_argument(arg: str) -> Tuple[str, str]:
    """
    Parse dataset argument in the form LABEL=PATH.
    """
    if "=" not in arg:
        raise ValueError(f"Dataset specification '{arg}' must be LABEL=PATH")
    label, path = arg.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Dataset specification '{arg}' is invalid")
    return label, path


def assign_style(label: str, style_state: Dict[str, object]) -> Dict[str, str]:
    """
    Assign a color/style dictionary for a dataset label.
    """
    used_presets = style_state.setdefault("used_presets", set())
    fallback_index = style_state.setdefault("fallback_index", 0)

    if label in STYLE_PRESETS:
        used_presets.add(label)
        return STYLE_PRESETS[label]

    for preset_label, preset in STYLE_PRESETS.items():
        if preset_label not in used_presets:
            used_presets.add(preset_label)
            return preset

    if not FALLBACK_STYLES:
        raise RuntimeError("Ran out of predefined styles for datasets.")
    style = FALLBACK_STYLES[fallback_index % len(FALLBACK_STYLES)]
    style_state["fallback_index"] = fallback_index + 1
    return style


def calculate_shared_direction_for_subset(datasets: List[Dict], subset_size: int) -> Dict:
    """
    Calculate a shared direction across all datasets for a given subset size.
    """
    subset_points = []
    centroids = {}
    for ds in datasets:
        coords = ds["points"][:subset_size, :3]
        centroid = np.mean(coords, axis=0)
        centroids[ds["label"]] = centroid
        subset_points.append(coords - centroid)

    combined_centered = np.vstack(subset_points)
    if combined_centered.shape[0] < 2:
        return {}

    pca = PCA(n_components=1)
    pca.fit(combined_centered)
    direction = pca.components_[0]
    if direction[0] < 0:
        direction = -direction

    dataset_equations = {}
    for label, centroid in centroids.items():
        dataset_equations[label] = {
            "centroid": centroid.tolist(),
            "parametric_equation": (
                f"p_sym = {centroid[0]:.4f} + t*{direction[0]:.4f}, "
                f"p_asym = {centroid[1]:.4f} + t*{direction[1]:.4f}, "
                f"p_dif = {centroid[2]:.4f} + t*{direction[2]:.4f}"
            )
        }

    return {
        "n_points": subset_size,
        "direction": direction.tolist(),
        "datasets": dataset_equations
    }


def calculate_shared_directions(datasets: List[Dict], subset_sizes: List[int], n_cpus: int) -> Dict[str, Dict]:
    """
    Calculate shared directions for multiple subset sizes in parallel.
    """
    tasks = [n for n in subset_sizes if all(len(ds["points"]) >= n for ds in datasets)]
    if not tasks:
        return {}

    with Pool(processes=n_cpus) as pool:
        results = pool.map(partial(calculate_shared_direction_for_subset, datasets), tasks)

    shared = {}
    for result in results:
        if result:
            shared[str(result["n_points"])] = result
    return shared


def print_summary_statistics(datasets: List[Dict]):
    """
    Print summary statistics for each dataset.
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for ds in datasets:
        points = ds["points"][:100]
        print(f"\n{ds['label']} Dataset (up to 100 lowest loss points):")
        print(f"  Count: {len(points)}")
        if len(points) == 0:
            continue
        print(f"  Loss range: {points[0, 3]:.6f} - {points[-1, 3]:.6f}")
        print(f"  Mean loss: {np.mean(points[:, 3]):.6f}")
        print(f"  Median loss: {np.median(points[:, 3]):.6f}")
        print(
            f"  Best point: p_sym={points[0, 0]:.4f}, "
            f"p_asym={points[0, 1]:.4f}, "
            f"p_dif={points[0, 2]:.4f}"
        )
        print("  Parameter ranges:")
        print(f"    p_sym: [{np.min(points[:, 0]):.4f}, {np.max(points[:, 0]):.4f}]")
        print(f"    p_asym: [{np.min(points[:, 1]):.4f}, {np.max(points[:, 1]):.4f}]")
        print(f"    p_dif: [{np.min(points[:, 2]):.4f}, {np.max(points[:, 2]):.4f}]")
    print("\n" + "=" * 60)


def build_annotation_text(datasets: List[Dict], dataset_lines: Dict[str, Dict], shared_directions: Dict[str, Dict], default_subset: int) -> str:
    """
    Build HTML annotation summarizing line equations for the default subset.
    """
    parts = []
    subset_key = str(default_subset)
    for ds in datasets:
        label = ds["label"]
        style = ds["style"]
        line_info = dataset_lines.get(label, {}).get(subset_key)
        if line_info:
            parts.append(
                f"<span style='color:{style['line_color']}'><b>{label} Line:</b> "
                f"{line_info['parametric_equation']}</span>"
            )
        shared_info = shared_directions.get(subset_key, {}).get("datasets", {}).get(label)
        if shared_info:
            parts.append(
                f"<span style='color:{style['shared_color']}'><b>{label} Shared:</b> "
                f"{shared_info['parametric_equation']}</span>"
            )
    parts.append("<span style='color:gray; font-size:12px'>p = [p_sym, p_asym, p_dif] = [centroid] + t·[direction]</span>")
    return "<br>".join(parts)


def create_interactive_3d_plot(datasets: List[Dict], subset_sizes: List[int], dataset_lines: Dict[str, Dict], shared_directions: Dict[str, Dict], output_path: str):
    """
    Create the interactive 3D plot with slider controls.
    """
    fig = go.Figure()

    # Legend stubs to keep entries even when traces hidden
    for ds in datasets:
        label = ds["label"]
        style = ds["style"]
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", name=label,
            marker=dict(size=10, color=style["marker_color"], opacity=0.8),
            showlegend=True, visible="legendonly", legendgroup=f"{label}_points"
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="lines", name=f"{label} Line",
            line=dict(color=style["line_color"], width=8),
            showlegend=True, visible="legendonly", legendgroup=f"{label}_line"
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="lines", name=f"{label} Shared",
            line=dict(color=style["shared_color"], width=8, dash="dash"),
            showlegend=True, visible="legendonly", legendgroup=f"{label}_shared"
        ))

    legend_stub_count = len(fig.data)

    trace_groups: List[List[int]] = []
    for subset in subset_sizes:
        indices_for_subset = []
        subset_key = str(subset)
        for ds in datasets:
            label = ds["label"]
            style = ds["style"]
            points = ds["points"]
            if len(points) < subset:
                continue

            subset_points = points[:subset]
            point_sizes = 15 * (1.0 / (1.0 + 100 * subset_points[:, 3]))
            trace_idx = len(fig.data)
            fig.add_trace(go.Scatter3d(
                x=subset_points[:, 0],
                y=subset_points[:, 1],
                z=subset_points[:, 2],
                mode="markers",
                name=label,
                marker=dict(
                    size=point_sizes,
                    color=subset_points[:, 3],
                    colorscale=style["colorscale"],
                    cmin=subset_points[:, 3].min(),
                    cmax=subset_points[:, 3].max(),
                    showscale=False,
                    opacity=0.8,
                    line=dict(width=0.5, color=style["marker_color"])
                ),
                text=[
                    f'{label} Point {i+1}<br>'
                    f'p_sym: {p[0]:.4f}<br>'
                    f'p_asym: {p[1]:.4f}<br>'
                    f'p_dif: {p[2]:.4f}<br>'
                    f'Loss: {p[3]:.6f}'
                    for i, p in enumerate(subset_points)
                ],
                hovertemplate="%{text}<extra></extra>",
                legendgroup=f"{label}_points",
                showlegend=False,
                visible=False
            ))
            indices_for_subset.append(trace_idx)

            line_info = dataset_lines.get(label, {}).get(subset_key)
            if line_info:
                x_line, y_line, z_line = create_line_points(line_info["line_params"])
                if len(x_line) > 0:
                    trace_idx = len(fig.data)
                    fig.add_trace(go.Scatter3d(
                        x=x_line, y=y_line, z=z_line,
                        mode="lines",
                        name=f"{label} Line",
                        line=dict(color=style["line_color"], width=8),
                        legendgroup=f"{label}_line",
                        showlegend=False,
                        visible=False,
                        hovertemplate=f"{line_info['parametric_equation']}<extra></extra>"
                    ))
                    indices_for_subset.append(trace_idx)

            shared_info = shared_directions.get(subset_key, {})
            dataset_shared = shared_info.get("datasets", {}).get(label)
            if dataset_shared:
                line_params = {
                    "centroid": dataset_shared["centroid"],
                    "direction": shared_info["direction"]
                }
                x_shared, y_shared, z_shared = create_line_points(line_params)
                if len(x_shared) > 0:
                    trace_idx = len(fig.data)
                    fig.add_trace(go.Scatter3d(
                        x=x_shared, y=y_shared, z=z_shared,
                        mode="lines",
                        name=f"{label} Shared",
                        line=dict(color=style["shared_color"], width=8, dash="dash"),
                        legendgroup=f"{label}_shared",
                        showlegend=False,
                        visible=False,
                        hovertemplate=f"{dataset_shared['parametric_equation']}<extra></extra>"
                    ))
                    indices_for_subset.append(trace_idx)

        trace_groups.append(indices_for_subset)

    # Slider configuration
    slider_steps = []
    total_traces = len(fig.data)
    for subset, indices in zip(subset_sizes, trace_groups):
        visibility = ['legendonly'] * legend_stub_count + [False] * (total_traces - legend_stub_count)
        for idx in indices:
            visibility[idx] = True
        slider_steps.append({
            "label": str(subset),
            "method": "update",
            "args": [{"visible": visibility}]
        })

    default_subset = 100
    if default_subset not in subset_sizes and subset_sizes:
        default_subset = subset_sizes[len(subset_sizes) // 2]
    default_step_index = subset_sizes.index(default_subset) if subset_sizes else 0

    annotation_text = build_annotation_text(datasets, dataset_lines, shared_directions, default_subset) if subset_sizes else "Line equations will appear here."

    fig.update_layout(
        title=f"Lowest Loss Points (Showing {default_subset} points per dataset)" if subset_sizes else "Lowest Loss Points",
        sliders=[{
            "active": default_step_index,
            "currentvalue": {"prefix": "Points per dataset: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }],
        showlegend=True,
        scene=dict(
            xaxis=dict(title="p_sym", range=[0, 0.6]),
            yaxis=dict(title="p_asym", range=[0, 0.4]),
            zaxis=dict(title="p_dif", range=[0, 0.4]),
            aspectmode="cube"
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=60, t=120),
        font=dict(family="Arial, sans-serif", size=12),
        annotations=[{
            "text": annotation_text,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 1.08,
            "xanchor": "center",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 14, "family": "monospace"},
            "align": "center",
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "gray",
            "borderwidth": 1,
            "borderpad": 4
        }]
    )

    # Apply default visibility
    if slider_steps:
        default_visibility = slider_steps[default_step_index]["args"][0]["visible"]
        for idx, vis in enumerate(default_visibility):
            fig.data[idx].visible = vis

    fig.write_html(output_path)
    print(f"Interactive plot saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create interactive 3D plot of lowest loss points for multiple datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset specification in the form LABEL=PATH. Provide once per dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for HTML file"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=1000,
        help="Maximum number of lowest loss points to extract per dataset"
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=32,
        help="Number of CPUs to use for parallel computations"
    )
    parser.add_argument(
        "--subset-min",
        type=int,
        default=50,
        help="Minimum number of points per dataset to include in the slider (default: 50)"
    )
    parser.add_argument(
        "--subset-step",
        type=int,
        default=10,
        help="Step size between slider values (default: 10)"
    )
    parser.add_argument(
        "--subset-max",
        type=int,
        default=500,
        help="Maximum number of points per dataset to include in the slider (default: 500)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = []
    style_state: Dict[str, object] = {}
    for spec in args.dataset:
        label, path = parse_dataset_argument(spec)
        style = assign_style(label, style_state)
        print(f"Loading dataset '{label}' from {path}")
        points = load_and_extract_lowest_points(path, args.n_points)
        print(f"  Extracted {len(points)} points")
        datasets.append({
            "label": label,
            "path": path,
            "points": points,
            "style": style
        })

    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for comparison.")

    min_points = min(len(ds["points"]) for ds in datasets)
    subset_min = max(1, args.subset_min)
    subset_step = max(1, args.subset_step)
    subset_max = max(subset_min, args.subset_max)

    available_max = min(min_points, subset_max)
    if available_max < subset_min:
        subset_min = max(1, available_max)

    if available_max < subset_min:
        raise ValueError("Not enough points to satisfy the requested subset minimum.")

    subset_sizes = list(range(subset_min, available_max + 1, subset_step))
    if subset_sizes[-1] != available_max:
        subset_sizes.append(available_max)

    subset_sizes = sorted(set(subset_sizes))

    print(f"\nCalculating line fits using up to {args.n_cpus} CPUs...")
    dataset_lines = {}
    for ds in datasets:
        print(f"  Calculating line fits for {ds['label']}...")
        dataset_lines[ds["label"]] = calculate_line_fits_parallel(ds["points"], subset_sizes, args.n_cpus)

    print("  Calculating shared directions across datasets...")
    shared_directions = calculate_shared_directions(datasets, subset_sizes, args.n_cpus)

    print_summary_statistics(datasets)
    create_interactive_3d_plot(datasets, subset_sizes, dataset_lines, shared_directions, args.output)


if __name__ == "__main__":
    main()
