#!/usr/bin/env python
"""Generate Figure 1b CD44/CD62L activation flow cytometry from raw FCS data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


FCS_DIR = (
    "Figure1_data_handoff/02_activation_flow/"
    "fcs_raw_2025-05-17_to_2025-05-23"
)
FLOWJO_WORKSPACE = (
    "Figure1_data_handoff/02_activation_flow/"
    "plot_exports_and_workspace/20-May-2025.wsp"
)
OUTPUT_DIR = "manuscript/plots/Figure1"
PANEL_STEM = "b_flow_activation_phenotype"

CD44_NAME = "FL1-A"
CD62L_NAME = "FL11-A"
CD44_LABEL = "CD44 PB450"
CD62L_LABEL = "CD62L APC"

ACTIVE_COLOR = "#ff9e9e"
NONACTIVE_COLOR = "#70b8ff"
GATE_COLOR = "#2f2f2f"
CONTOUR_SMOOTHING_SIGMA = 8.0
DPI = 300


@dataclass(frozen=True)
class OverlaySample:
    label: str
    filename: str
    color: str


@dataclass(frozen=True)
class PanelSpec:
    day_label: str
    row_label: str
    samples: tuple[OverlaySample, OverlaySample]


@dataclass(frozen=True)
class FlowGate:
    lymphocyte_vertices: np.ndarray
    cd8_ranges: dict[str, tuple[float | None, float | None]]
    cd44_threshold: float
    cd62l_threshold: float


@dataclass(frozen=True)
class FlowJoBiexTransform:
    max_value: float
    negative: float
    width: float
    positive: float


FLOWJO_BIEX_TRANSFORMS = {
    CD44_NAME: FlowJoBiexTransform(
        max_value=1677721.5999999999,
        negative=0.0,
        width=-15.8489319246,
        positive=4.6,
    ),
    CD62L_NAME: FlowJoBiexTransform(
        max_value=530542.1535661949,
        negative=0.0,
        width=-100.0,
        positive=4.7247198959,
    ),
}


PANEL_GRID = [
    [
        PanelSpec(
            day_label="3 days",
            row_label="Non-transduced",
            samples=(
                OverlaySample(
                    "Non-activated",
                    "Non-transduced Non-activated full panel.fcs",
                    NONACTIVE_COLOR,
                ),
                OverlaySample(
                    "Activated",
                    "Non-transduced Activated full panel.fcs",
                    ACTIVE_COLOR,
                ),
            ),
        ),
        PanelSpec(
            day_label="5 days",
            row_label="Non-transduced",
            samples=(
                OverlaySample(
                    "Non-activated",
                    "Non-transduced Non-activated full panel 5_19.fcs",
                    NONACTIVE_COLOR,
                ),
                OverlaySample(
                    "Activated",
                    "Non-transduced Activated full panel 5_19.fcs",
                    ACTIVE_COLOR,
                ),
            ),
        ),
    ],
    [
        PanelSpec(
            day_label="3 days",
            row_label="Transduced",
            samples=(
                OverlaySample(
                    "Non-activated",
                    "Transduced Non-activated full panel 5_19.fcs",
                    NONACTIVE_COLOR,
                ),
                OverlaySample(
                    "Activated",
                    "Transduced Activated full panel 5_19.fcs",
                    ACTIVE_COLOR,
                ),
            ),
        ),
        PanelSpec(
            day_label="5 days",
            row_label="Transduced",
            samples=(
                OverlaySample(
                    "Non-activated",
                    "Transduced Non-activated full panel 5_23.fcs",
                    NONACTIVE_COLOR,
                ),
                OverlaySample(
                    "Activated",
                    "Transduced Activated full panel 5_23.fcs",
                    ACTIVE_COLOR,
                ),
            ),
        ),
    ],
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def namespaced_attribute(element: ET.Element, namespace: str, key: str) -> str | None:
    return element.attrib.get(f"{{{namespace}}}{key}") or element.attrib.get(key)


def child_populations(node: ET.Element) -> list[ET.Element]:
    populations = []
    for subpopulations in node.findall("Subpopulations"):
        populations.extend(subpopulations.findall("Population"))
    return populations


def find_population(node: ET.Element, name: str) -> ET.Element:
    for population in child_populations(node):
        if population.attrib.get("name") == name:
            return population
    raise ValueError(f"Could not find population {name}")


def parse_gate_dimensions(
    population: ET.Element,
) -> dict[str, tuple[float | None, float | None]]:
    gate = population.find("Gate")
    if gate is None:
        return {}

    ranges = {}
    for element in gate.iter():
        if local_name(element.tag) != "dimension":
            continue
        channel = None
        for child in element:
            if local_name(child.tag) == "fcs-dimension":
                channel = namespaced_attribute(
                    child,
                    "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
                    "name",
                )
        if channel is None:
            continue
        minimum = namespaced_attribute(
            element,
            "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
            "min",
        )
        maximum = namespaced_attribute(
            element,
            "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
            "max",
        )
        ranges[channel] = (
            float(minimum) if minimum is not None else None,
            float(maximum) if maximum is not None else None,
        )
    return ranges


def parse_polygon_vertices(population: ET.Element) -> np.ndarray:
    gate = population.find("Gate")
    if gate is None:
        raise ValueError(f"{population.attrib.get('name')} has no gate")

    vertices = []
    for element in gate.iter():
        if local_name(element.tag) != "vertex":
            continue
        coordinates = []
        for child in element:
            if local_name(child.tag) != "coordinate":
                continue
            value = namespaced_attribute(
                child,
                "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
                "value",
            )
            coordinates.append(float(value))
        vertices.append(coordinates)
    return np.asarray(vertices, dtype=float)


def parse_workspace_gates(workspace_path: Path) -> dict[str, FlowGate]:
    root = ET.parse(workspace_path).getroot()
    sample_names = {
        sample.filename
        for row in PANEL_GRID
        for panel in row
        for sample in panel.samples
    }
    gates = {}

    for sample_node in root.iter("SampleNode"):
        sample_name = sample_node.attrib.get("name")
        if sample_name not in sample_names:
            continue

        lymphocytes = find_population(sample_node, "Lymphocytes")
        cd8_cells = find_population(lymphocytes, "CD8 T cells")
        q1 = find_population(cd8_cells, "Q1: CD44 PB450-A- , CD62L APC-A+")
        q2 = find_population(cd8_cells, "Q2: CD44 PB450-A+ , CD62L APC-A+")
        q1_ranges = parse_gate_dimensions(q1)
        q2_ranges = parse_gate_dimensions(q2)

        cd44_threshold = q2_ranges[CD44_NAME][0]
        cd62l_threshold = q1_ranges[CD62L_NAME][0]
        if cd44_threshold is None or cd62l_threshold is None:
            raise ValueError(f"Missing CD44/CD62L thresholds for {sample_name}")

        gates[sample_name] = FlowGate(
            lymphocyte_vertices=parse_polygon_vertices(lymphocytes),
            cd8_ranges=parse_gate_dimensions(cd8_cells),
            cd44_threshold=cd44_threshold,
            cd62l_threshold=cd62l_threshold,
        )

    missing = sorted(sample_names - set(gates))
    if missing:
        raise ValueError(f"Missing FlowJo gates for samples: {missing}")
    return gates


def parse_fcs_text(text: str) -> dict[str, str]:
    delimiter = text[0]
    parts = text[1:].split(delimiter)
    return dict(zip(parts[0::2], parts[1::2]))


def read_fcs(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with path.open("rb") as handle:
        header = handle.read(58)
        version = header[:6].decode("ascii", errors="replace")
        if not version.startswith("FCS3"):
            raise ValueError(f"{path} is not an FCS3 file: {version}")

        text_start = int(header[10:18])
        text_end = int(header[18:26])
        data_start = int(header[26:34])
        data_end = int(header[34:42])
        handle.seek(text_start)
        text = handle.read(text_end - text_start + 1).decode("latin1")
        meta = parse_fcs_text(text)

        if meta.get("$DATATYPE") != "F":
            raise ValueError(f"{path} uses unsupported FCS datatype {meta.get('$DATATYPE')}")
        if meta.get("$BYTEORD") != "1,2,3,4":
            raise ValueError(f"{path} uses unsupported byte order {meta.get('$BYTEORD')}")

        n_events = int(meta["$TOT"])
        n_parameters = int(meta["$PAR"])
        handle.seek(data_start)
        data = np.frombuffer(
            handle.read(data_end - data_start + 1),
            dtype="<f4",
            count=n_events * n_parameters,
        ).reshape(n_events, n_parameters)

    columns = [
        meta.get(f"$P{i}S") or meta.get(f"$P{i}N") or f"P{i}"
        for i in range(1, n_parameters + 1)
    ]
    channel_map = {
        meta.get(f"$P{i}N"): columns[i - 1]
        for i in range(1, n_parameters + 1)
        if meta.get(f"$P{i}N") is not None
    }
    return pd.DataFrame(data, columns=columns), channel_map


def apply_range_gate(
    values: np.ndarray,
    minimum: float | None,
    maximum: float | None,
) -> np.ndarray:
    keep = np.ones(values.shape[0], dtype=bool)
    if minimum is not None:
        keep &= values >= minimum
    if maximum is not None:
        keep &= values <= maximum
    return keep


def apply_flowjo_gate(
    data: pd.DataFrame,
    channel_map: dict[str, str],
    gate: FlowGate,
) -> pd.DataFrame:
    lymphocyte_path = MplPath(gate.lymphocyte_vertices)
    lymphocyte_columns = [channel_map["FSC-A"], channel_map["SSC-A"]]
    keep = lymphocyte_path.contains_points(data[lymphocyte_columns].to_numpy())

    for channel, (minimum, maximum) in gate.cd8_ranges.items():
        column = channel_map[channel]
        keep &= apply_range_gate(data[column].to_numpy(), minimum, maximum)

    return data.loc[keep].copy()


def _log_root(b: float, width: float) -> float:
    if width == 0:
        return b

    x_lo = 0.0
    x_hi = b
    d = (x_lo + x_hi) / 2
    dx = abs(int(x_lo - x_hi))
    dx_last = dx
    fb = -2 * np.log(b) + width * b
    f = 2.0 * np.log(d) + width * b + fb
    df = 2 / d + width

    for _ in range(100):
        if (
            (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0
            or abs(2 * f) > abs(dx_last * df)
        ):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            previous = d
            d -= dx
            if d == previous:
                return d

        if abs(dx) < 1.0e-12:
            return d

        dx_last = dx
        f = 2 * np.log(d) + width * d + fb
        df = 2 / d + width
        if f < 0:
            x_lo = d
        else:
            x_hi = d

    return d


def generate_flowjo_biex_lut(
    channel_range: int = 4096,
    positive: float = 4.418540,
    negative: float = 0.0,
    width_basis: float = -10.0,
    max_value: float = 262144.000029,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a FlowJo-compatible biexponential input-output lookup table."""
    ln10 = np.log(10.0)
    decades = positive
    width = np.log10(-width_basis)

    decades = decades - (width / 2)
    extra = max(negative, 0.0) + (width / 2)

    zero_point = int((extra * channel_range) / (extra + decades))
    zero_point = int(np.min([zero_point, channel_range / 2]))
    if zero_point > 0:
        decades = extra * channel_range / zero_point

    width = width / (2 * decades)
    positive_range = ln10 * decades
    minimum = max_value / np.exp(positive_range)
    negative_range = _log_root(positive_range, width)

    n_points = channel_range + 1
    values = np.arange(n_points, dtype=float)
    positive_values = np.exp(values / n_points * positive_range)
    negative_values = np.exp(values / n_points * -negative_range)

    scale = np.exp((positive_range + negative_range) * (width + extra / decades))
    negative_values *= scale
    offset = positive_values[zero_point] - negative_values[zero_point]

    positive_values[zero_point:n_points] -= negative_values[zero_point:n_points]
    positive_values[zero_point:n_points] = minimum * (
        positive_values[zero_point:n_points] - offset
    )

    negative_indices = np.arange(zero_point)
    mirror_indices = 2 * zero_point - negative_indices
    positive_values[negative_indices] = -positive_values[mirror_indices]

    return positive_values, values


@lru_cache(maxsize=None)
def flowjo_biex_lut(channel: str) -> tuple[np.ndarray, np.ndarray]:
    transform = FLOWJO_BIEX_TRANSFORMS[channel]
    return generate_flowjo_biex_lut(
        positive=transform.positive,
        negative=transform.negative,
        width_basis=transform.width,
        max_value=transform.max_value,
    )


def flow_axis_transform(values: np.ndarray | float, channel: str) -> np.ndarray | float:
    raw_values = np.asarray(values, dtype=float)
    input_values, output_values = flowjo_biex_lut(channel)
    transformed = np.interp(
        raw_values,
        input_values,
        output_values,
        left=output_values[0],
        right=output_values[-1],
    )
    if np.isscalar(values):
        return float(transformed)
    return transformed


def flow_display_limits() -> tuple[tuple[float, float], tuple[float, float]]:
    return (0.0, 4096.0), (0.0, 4096.0)


def flow_tick_label(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0:
        return r"$-10^3$"
    exponent = int(np.log10(value))
    return rf"$10^{exponent}$"


def contour_levels(density: np.ndarray, step: float = 0.05) -> np.ndarray:
    positive = np.sort(density[density > 0])[::-1]
    if positive.size == 0:
        return np.array([])
    cumulative_mass = np.cumsum(positive) / positive.sum()
    thresholds = []
    for mass_fraction in np.arange(step, 1.0, step):
        index = np.searchsorted(cumulative_mass, mass_fraction)
        index = min(index, positive.size - 1)
        thresholds.append(positive[index])
    levels = np.unique(np.asarray(thresholds, dtype=float))
    levels = levels[np.isfinite(levels)]
    if levels.size < 2:
        return np.array([])
    return np.sort(levels)


def draw_contours(
    ax: plt.Axes,
    data: pd.DataFrame,
    channel_map: dict[str, str],
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x = flow_axis_transform(data[channel_map[CD44_NAME]].to_numpy(), CD44_NAME)
    y = flow_axis_transform(data[channel_map[CD62L_NAME]].to_numpy(), CD62L_NAME)
    keep = np.isfinite(x) & np.isfinite(y)
    hist, x_edges, y_edges = np.histogram2d(
        x[keep],
        y[keep],
        bins=256,
        range=[xlim, ylim],
    )
    density = gaussian_filter(hist.T, sigma=CONTOUR_SMOOTHING_SIGMA)
    levels = contour_levels(density)
    if levels.size < 2:
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    contour = ax.contour(
        x_centers,
        y_centers,
        density,
        levels=levels,
        colors=color,
        linewidths=0.45,
        alpha=0.95,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Rasterization of .*QuadContourSet.* will be ignored",
            category=UserWarning,
        )
        contour.set_rasterized(True)


def style_flow_axis(ax: plt.Axes) -> None:
    raw_x_ticks = np.array([0, 1e4, 1e5, 1e6])
    raw_y_ticks = np.array([-1e3, 0, 1e3, 1e4, 1e5])
    ax.set_xticks(flow_axis_transform(raw_x_ticks, CD44_NAME))
    ax.set_yticks(flow_axis_transform(raw_y_ticks, CD62L_NAME))
    ax.set_xticklabels([flow_tick_label(value) for value in raw_x_ticks], fontsize=5)
    ax.set_yticklabels([flow_tick_label(value) for value in raw_y_ticks], fontsize=5)
    ax.tick_params(axis="both", which="both", width=0.7, length=3, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)


def sample_percentages(
    data: pd.DataFrame,
    channel_map: dict[str, str],
    gate: FlowGate,
) -> dict[str, float]:
    counts = gate_counts(data, channel_map, gate)
    n_cd8 = counts["n_cd8_gated_events"]
    return {
        "n_cd8_gated_events": float(n_cd8),
        "nonactivated_pct": 100 * counts["nonactivated_count"] / n_cd8,
        "activated_pct": 100 * counts["activated_count"] / n_cd8,
        "cd44_high_cd62l_low_pct": 100 * counts["cd44_high_cd62l_low_count"] / n_cd8,
    }


def gate_counts(
    data: pd.DataFrame,
    channel_map: dict[str, str],
    gate: FlowGate,
) -> dict[str, int]:
    cd44 = data[channel_map[CD44_NAME]].to_numpy()
    cd62l = data[channel_map[CD62L_NAME]].to_numpy()
    n_cd8 = data.shape[0]
    nonactivated = (cd44 < gate.cd44_threshold) & (cd62l >= gate.cd62l_threshold)
    activated = cd44 >= gate.cd44_threshold
    cd44_high_cd62l_low = (cd44 >= gate.cd44_threshold) & (cd62l < gate.cd62l_threshold)
    return {
        "n_cd8_gated_events": int(n_cd8),
        "nonactivated_count": int(nonactivated.sum()),
        "activated_count": int(activated.sum()),
        "cd44_high_cd62l_low_count": int(cd44_high_cd62l_low.sum()),
    }


def combined_panel_counts(
    panel: PanelSpec,
    loaded: dict[str, pd.DataFrame],
    channel_maps: dict[str, dict[str, str]],
    gates: dict[str, FlowGate],
) -> dict[str, int]:
    total = 0
    nonactivated_count = 0
    activated_count = 0
    for sample in panel.samples:
        counts = gate_counts(
            loaded[sample.filename],
            channel_maps[sample.filename],
            gates[sample.filename],
        )
        total += counts["n_cd8_gated_events"]
        nonactivated_count += counts["nonactivated_count"]
        activated_count += counts["activated_count"]
    return {
        "n_cd8_gated_events": total,
        "nonactivated_count": nonactivated_count,
        "activated_count": activated_count,
        "bottom_left_count": total - nonactivated_count - activated_count,
    }


def combined_panel_percentages(
    panel: PanelSpec,
    loaded: dict[str, pd.DataFrame],
    channel_maps: dict[str, dict[str, str]],
    gates: dict[str, FlowGate],
) -> dict[str, float]:
    counts = combined_panel_counts(panel, loaded, channel_maps, gates)
    total = counts["n_cd8_gated_events"]
    return {
        "nonactivated_pct": 100 * counts["nonactivated_count"] / total,
        "activated_pct": 100 * counts["activated_count"] / total,
        "bottom_left_pct": 100 * counts["bottom_left_count"] / total,
    }


def panel_gate_stats_table(
    loaded: dict[str, pd.DataFrame],
    channel_maps: dict[str, dict[str, str]],
    gates: dict[str, FlowGate],
) -> pd.DataFrame:
    rows = []
    for panel_row in PANEL_GRID:
        for panel in panel_row:
            counts = combined_panel_counts(panel, loaded, channel_maps, gates)
            total = counts["n_cd8_gated_events"]
            rows.append(
                {
                    "row_label": panel.row_label,
                    "day_label": panel.day_label,
                    **counts,
                    "nonactivated_pct": 100 * counts["nonactivated_count"] / total,
                    "activated_pct": 100 * counts["activated_count"] / total,
                    "bottom_left_pct": 100 * counts["bottom_left_count"] / total,
                }
            )
    return pd.DataFrame(rows)


def load_data(
    fcs_dir: Path,
    gates: dict[str, FlowGate],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], pd.DataFrame]:
    loaded = {}
    channel_maps = {}
    stats_rows = []

    filenames = []
    for row in PANEL_GRID:
        for panel in row:
            for sample in panel.samples:
                if sample.filename not in filenames:
                    filenames.append(sample.filename)

    for filename in filenames:
        data, channel_map = read_fcs(fcs_dir / filename)
        gated_data = apply_flowjo_gate(data, channel_map, gates[filename])
        loaded[filename] = gated_data
        channel_maps[filename] = channel_map
        percentages = sample_percentages(gated_data, channel_map, gates[filename])
        stats_rows.append(
            {
                "filename": filename,
                "cd44_threshold": gates[filename].cd44_threshold,
                "cd62l_threshold": gates[filename].cd62l_threshold,
                **percentages,
            }
        )

    return loaded, channel_maps, pd.DataFrame(stats_rows)


def draw_gate_boxes(
    ax: plt.Axes,
    gate: FlowGate,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_threshold = flow_axis_transform(gate.cd44_threshold, CD44_NAME)
    y_threshold = flow_axis_transform(gate.cd62l_threshold, CD62L_NAME)

    nonactive_box = Rectangle(
        (x_min, y_threshold),
        x_threshold - x_min,
        y_max - y_threshold,
        fill=False,
        edgecolor=GATE_COLOR,
        linewidth=0.75,
        linestyle=(0, (2.2, 1.8)),
    )
    active_box = Rectangle(
        (x_threshold, y_min),
        x_max - x_threshold,
        y_max - y_min,
        fill=False,
        edgecolor=GATE_COLOR,
        linewidth=0.75,
        linestyle=(0, (2.2, 1.8)),
    )
    ax.add_patch(nonactive_box)
    ax.add_patch(active_box)


def add_gate_percentages(
    ax: plt.Axes,
    nonactive_pct: float,
    active_pct: float,
) -> None:
    ax.text(
        0.05,
        0.92,
        f"Non-act {nonactive_pct:.0f}%",
        transform=ax.transAxes,
        color=NONACTIVE_COLOR,
        fontsize=5.4,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
    )
    ax.text(
        0.95,
        0.92,
        f"Act {active_pct:.0f}%",
        transform=ax.transAxes,
        color=ACTIVE_COLOR,
        fontsize=5.4,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
    )


def add_shared_axis_arrows(fig: plt.Figure) -> None:
    origin_x = 0.205
    origin_y = 0.09
    arrow_length_inches = 0.53
    fig_width, fig_height = fig.get_size_inches()
    arrow_dx = arrow_length_inches / fig_width
    arrow_dy = arrow_length_inches / fig_height

    fig.text(
        origin_x + arrow_dx * 0.78,
        0.055,
        CD44_LABEL,
        fontsize=12,
        ha="center",
        va="center",
    )
    fig.text(0.185, 0.16, CD62L_LABEL, fontsize=12, ha="center", va="center", rotation=90)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    arrowprops = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 1.8,
        "mutation_scale": 11,
    }
    ax.annotate(
        "",
        xy=(origin_x, origin_y + arrow_dy),
        xytext=(origin_x, origin_y),
        xycoords=fig.transFigure,
        arrowprops=arrowprops,
    )
    ax.annotate(
        "",
        xy=(origin_x + arrow_dx, origin_y),
        xytext=(origin_x, origin_y),
        xycoords=fig.transFigure,
        arrowprops=arrowprops,
    )


def make_figure(
    loaded: dict[str, pd.DataFrame],
    channel_maps: dict[str, dict[str, str]],
    gates: dict[str, FlowGate],
) -> plt.Figure:
    xlim, ylim = flow_display_limits()

    fig = plt.figure(figsize=(7.8, 5.15))
    axis_width = 0.18
    axis_height = axis_width * 7.8 / 5.15
    left_positions = [0.245, 0.49]
    bottom_positions = [0.47, 0.15]

    for row_idx, panel_row in enumerate(PANEL_GRID):
        for col_idx, panel in enumerate(panel_row):
            ax = fig.add_axes(
                [
                    left_positions[col_idx],
                    bottom_positions[row_idx],
                    axis_width,
                    axis_height,
                ]
            )
            for sample in panel.samples:
                draw_contours(
                    ax,
                    loaded[sample.filename],
                    channel_maps[sample.filename],
                    sample.color,
                    xlim,
                    ylim,
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            style_flow_axis(ax)
            draw_gate_boxes(ax, gates[panel.samples[0].filename], xlim, ylim)
            panel_stats = combined_panel_percentages(
                panel,
                loaded,
                channel_maps,
                gates,
            )
            add_gate_percentages(
                ax,
                panel_stats["nonactivated_pct"],
                panel_stats["activated_pct"],
            )
            if row_idx == 0:
                ax.set_title(panel.day_label, fontsize=16, pad=4)

    fig.text(
        0.455,
        0.815,
        "days post-CD3/CD28 stimulation",
        fontsize=17,
        ha="center",
        va="center",
    )
    fig.text(0.03, 0.59, "Non-transduced", fontsize=15, ha="left", va="center")
    fig.text(0.03, 0.27, "Transduced", fontsize=15, ha="left", va="center")

    legend_x = 0.735
    fig.add_artist(
        Rectangle(
            (legend_x, 0.64),
            0.018,
            0.024,
            transform=fig.transFigure,
            color=ACTIVE_COLOR,
            linewidth=0,
        )
    )
    fig.text(legend_x + 0.03, 0.651, "Activated", fontsize=17, ha="left", va="center")
    fig.add_artist(
        Rectangle(
            (legend_x, 0.59),
            0.018,
            0.024,
            transform=fig.transFigure,
            color=NONACTIVE_COLOR,
            linewidth=0,
        )
    )
    fig.text(
        legend_x + 0.03,
        0.601,
        "Non-activated",
        fontsize=17,
        ha="left",
        va="center",
    )

    add_shared_axis_arrows(fig)

    return fig


def save_sample_table(output_dir: Path) -> None:
    rows = []
    for panel_row in PANEL_GRID:
        for panel in panel_row:
            for sample in panel.samples:
                rows.append(
                    {
                        "day_label": panel.day_label,
                        "row_label": panel.row_label,
                        "overlay_label": sample.label,
                        "filename": sample.filename,
                    }
                )
    pd.DataFrame(rows).to_csv(
        output_dir / f"{PANEL_STEM}_samples.tsv",
        sep="\t",
        index=False,
    )


def save_outputs(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{PANEL_STEM}.png"
    svg_path = output_dir / f"{PANEL_STEM}.svg"
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    root = repo_root()
    output_dir = root / OUTPUT_DIR
    gates = parse_workspace_gates(root / FLOWJO_WORKSPACE)
    loaded, channel_maps, stats = load_data(root / FCS_DIR, gates)
    fig = make_figure(loaded, channel_maps, gates)
    save_outputs(fig, output_dir)
    save_sample_table(output_dir)
    stats.to_csv(output_dir / f"{PANEL_STEM}_gate_stats.tsv", sep="\t", index=False)
    panel_gate_stats_table(loaded, channel_maps, gates).to_csv(
        output_dir / f"{PANEL_STEM}_panel_gate_stats.tsv",
        sep="\t",
        index=False,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
