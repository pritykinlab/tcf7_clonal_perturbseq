#!/usr/bin/env python
"""Generate Figure 7 Trim28 functional validation panels from handoff tables."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.grid"] = False


PANEL_TABLES = {
    "a": "Fig7A_TCF7_validation_clean.csv",
    "b": "Fig7B_competitive_recovery_excluding_pd1_overlap_clean.csv",
    "c": "Fig7C_PD1_enrichment_plus_exp74_mannwhitney_clean.csv",
    "d": "Fig7C_PD1_enrichment_plus_exp74_mannwhitney_clean.csv",
    "d_experimental": "Fig7C_PD1_enrichment_plus_exp74_mannwhitney_clean.csv",
    "e": (
        "Fig7D_exp71_growth_no_pd1_forwardfilled_clean.csv",
        "Fig7D_exp71_survival_no_pd1_forwardfilled_clean.csv",
        "Fig7D_exp67_growth_pd1_clean.csv",
        "Fig7D_exp67_survival_pd1_clean.csv",
    ),
    "f": (
        "Fig7E_skin_distribution_3bin_clean.csv",
        "Fig7E_peritumor_vs_distant_pairs.csv",
    ),
}

def blend_color(color: str, target: str, fraction: float) -> str:
    src = to_rgb(color)
    dst = to_rgb(target)
    return to_hex(tuple((1.0 - fraction) * s + fraction * d for s, d in zip(src, dst)))


# Match scripts/utils/figure_format.py perturbation_colors without importing that
# scanpy-heavy module: NT is "grey"; Trim28 is plt.get_cmap("Paired")(7).
NT_COLOR = "grey"
TRIM28_COLOR = to_hex(plt.get_cmap("Paired")(7))
ISOTYPE_COLOR = "#FA8072"
APD1_COLOR = "#87CEFA"

COLORS = {
    "axis": "#202020",
    "grid": "#d8d8d8",
    "muted": "#6f6f6f",
    "light": "#c9c9c9",
    "nt": NT_COLOR,
    "nt_light": blend_color(NT_COLOR, "white", 0.50),
    "trim": TRIM28_COLOR,
    "trim_dark": blend_color(TRIM28_COLOR, "black", 0.30),
    "trim_light": blend_color(TRIM28_COLOR, "white", 0.72),
    "peritumor": "#7b7b7b",
    "isotype": ISOTYPE_COLOR,
    "apd1": APD1_COLOR,
}

DPI = 300
TWO_GROUP_FIGSIZE = (3.45, 4.0)
TWO_GROUP_X = (0.0, 0.70)
TWO_GROUP_XLIM = (-0.36, 1.06)
TWO_GROUP_JITTER = 0.16
TWO_GROUP_SUBPLOTS_ADJUST = {
    "left": 0.30,
    "right": 0.97,
    "bottom": 0.25,
    "top": 0.84,
}
GROWTH_SURVIVAL_TEXT_SCALE = 1.5
GROWTH_BACKGROUND_LINE_WIDTH = 0.9
GROWTH_BACKGROUND_LINE_ALPHA = 0.58
GROWTH_NT_BACKGROUND = blend_color(NT_COLOR, "white", 0.35)
GROWTH_TRIM28_BACKGROUND = blend_color(TRIM28_COLOR, "white", 0.52)
GROWTH_SURVIVAL_FONT_PARAMS = {
    "font.size": plt.rcParams["font.size"] * GROWTH_SURVIVAL_TEXT_SCALE,
    "axes.titlesize": plt.rcParams["axes.titlesize"] * GROWTH_SURVIVAL_TEXT_SCALE,
    "axes.labelsize": plt.rcParams["axes.labelsize"] * GROWTH_SURVIVAL_TEXT_SCALE,
    "xtick.labelsize": plt.rcParams["xtick.labelsize"] * GROWTH_SURVIVAL_TEXT_SCALE,
    "ytick.labelsize": plt.rcParams["ytick.labelsize"] * GROWTH_SURVIVAL_TEXT_SCALE,
    "legend.fontsize": plt.rcParams["legend.fontsize"] * GROWTH_SURVIVAL_TEXT_SCALE,
}
GROWTH_TREATMENT_TITLE_SIZE = GROWTH_SURVIVAL_FONT_PARAMS["axes.titlesize"] * 1.5
PD1_REQUIRED_COLUMNS = [
    "experiment",
    "sample",
    "source_type",
    "treatment_for_plot",
    "normalized_Trim28_TIL",
    "TIL_over_dLN_Trim28_ratio",
]
PD1_METRICS = {
    "c": ("normalized_Trim28_TIL", "Tumor\noutput% / input%"),
    "d": ("TIL_over_dLN_Trim28_ratio", "Tumor / dLN output%"),
}
SKIN_INPUT_TRIM28_FRACTIONS = {
    "Exp66": 0.523,
    "Exp74": 0.5352009921937696,
}
SKIN_INPUT_TRIM28_FRACTION_NOTES = {
    "Exp66": "Inferred from Figure 7C Exp66 normalized values: output Trim28 fraction / normalized value = 0.523.",
    "Exp74": "Computed directly from guide_counts2.xlsx input counts: 14672 Trim28 reads / (12742 NT + 14672 Trim28 reads).",
}
SKIN_EXCLUDED_EXPERIMENTS = {
    "Exp70": "Excluded because no explicit input denominator was found in the available source files.",
}
FIGURE7B_TIL_TRIM28_NORM_CORRECTIONS = {
    "exp61_": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=sorted(PANEL_TABLES), required=True)
    parser.add_argument(
        "--clean-dir",
        default=(
            "../Figure7_data_context_2026-05-19/02_cleaned_tables_and_outputs/"
            "figure7_redesign/cleaned_tables"
        ),
    )
    parser.add_argument(
        "--parsed-dir",
        default="../Figure7_data_context_2026-05-19/01_collected_source_data/figure7_data/parsed",
    )
    parser.add_argument("--output-dir", default="plots/Figure7")
    return parser.parse_args()


def stable_jitter(key: object, width: float) -> float:
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * width


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def exact_paired_signflip_p(diffs: list[float]) -> float:
    if not diffs:
        raise ValueError("Cannot run paired sign-flip test without paired differences.")
    observed = abs(sum(diffs))
    half = len(diffs) // 2
    left = diffs[:half]
    right = diffs[half:]

    left_sums = [0.0]
    for value in left:
        left_sums = [x + value for x in left_sums] + [x - value for x in left_sums]

    right_sums = [0.0]
    for value in right:
        right_sums = [x + value for x in right_sums] + [x - value for x in right_sums]
    right_sums.sort()

    extreme = 0
    for left_sum in left_sums:
        extreme += len(right_sums) - bisect.bisect_left(
            right_sums, observed - left_sum - 1e-12
        )
        extreme += bisect.bisect_right(right_sums, -observed - left_sum + 1e-12)
    return extreme / (2 ** len(diffs))


def normal_cdf(z_value: float) -> float:
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


def rank_values(values: list[float]) -> tuple[list[float], float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_correction = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        tie_len = j - i
        tie_correction += tie_len**3 - tie_len
        i = j
    return ranks, tie_correction


def wilcoxon_signed_rank_two_sided(diffs: list[float]) -> dict[str, float]:
    usable_diffs = [float(diff) for diff in diffs if float(diff) != 0.0]
    if not usable_diffs:
        raise ValueError("Wilcoxon signed-rank test requires at least one nonzero paired difference.")
    ranks, _tie_correction = rank_values([abs(diff) for diff in usable_diffs])
    w_positive = sum(rank for rank, diff in zip(ranks, usable_diffs) if diff > 0)
    w_negative = sum(rank for rank, diff in zip(ranks, usable_diffs) if diff < 0)
    statistic = min(w_positive, w_negative)

    scaled_ranks = [int(round(rank * 2)) for rank in ranks]
    observed = int(round(statistic * 2))
    counts = {0: 1}
    for rank in scaled_ranks:
        next_counts = counts.copy()
        for value, count in counts.items():
            next_counts[value + rank] = next_counts.get(value + rank, 0) + count
        counts = next_counts
    lower_tail = sum(count for value, count in counts.items() if value <= observed)
    p_value = min(1.0, 2.0 * lower_tail / (2 ** len(usable_diffs)))
    return {
        "n": len(usable_diffs),
        "zero_differences_removed": len(diffs) - len(usable_diffs),
        "w_positive": w_positive,
        "w_negative": w_negative,
        "statistic": statistic,
        "p": p_value,
    }


def mann_whitney_two_sided(control: list[float], pd1: list[float]) -> dict[str, float]:
    if not control or not pd1:
        raise ValueError("Mann-Whitney test requires values in both groups.")
    values = control + pd1
    ranks, tie_correction = rank_values(values)
    n_control = len(control)
    n_pd1 = len(pd1)
    rank_sum_control = sum(ranks[:n_control])
    u_control = rank_sum_control - n_control * (n_control + 1) / 2.0
    u_pd1 = n_control * n_pd1 - u_control
    n_total = n_control + n_pd1
    mu = n_control * n_pd1 / 2.0
    sigma = math.sqrt(
        n_control
        * n_pd1
        / 12.0
        * ((n_total + 1.0) - tie_correction / (n_total * (n_total - 1.0)))
    )
    u_min = min(u_control, u_pd1)
    z_value = (u_min - mu + 0.5) / sigma
    p_value = min(1.0, 2.0 * normal_cdf(z_value))
    return {
        "control_n": n_control,
        "anti_pd1_n": n_pd1,
        "control_mean": mean(control),
        "anti_pd1_mean": mean(pd1),
        "control_median": median(control),
        "anti_pd1_median": median(pd1),
        "u_control": u_control,
        "u_anti_pd1": u_pd1,
        "z": z_value,
        "p": p_value,
    }


def format_p(p_value: float | None) -> str:
    if p_value is None or math.isnan(p_value):
        return "p = n/a"
    if p_value < 0.0001:
        return "p < 0.0001"
    if p_value < 0.001:
        return f"p = {p_value:.4f}"
    return f"p = {p_value:.3f}"


def format_logrank_p(p_value: float | None) -> str:
    if p_value is None or math.isnan(p_value):
        return "p = n/a"
    if p_value < 0.0001:
        return "p < 0.0001"
    if p_value < 0.1:
        return f"p = {p_value:.3f}"
    return f"p = {p_value:.2f}"


def format_paired_p(p_value: float | None, n_pairs: int) -> str:
    return f"{format_p(p_value)}\npaired n={n_pairs}"


def format_axis_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def style_axis(ax: plt.Axes, y_ticks: list[float] | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3, color=COLORS["axis"])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)


def draw_boxplot(ax: plt.Axes, x_value: float, values: list[float], color: str) -> None:
    if not values:
        return
    box = ax.boxplot(
        [values],
        positions=[x_value],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        manage_ticks=False,
        boxprops={
            "facecolor": blend_color(color, "white", 0.65),
            "edgecolor": color,
            "linewidth": 1.2,
            "alpha": 1.0,
        },
        medianprops={"color": color, "linewidth": 1.6},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_zorder(1.5)
    for artist_group in ("whiskers", "caps", "medians"):
        for artist in box[artist_group]:
            artist.set_zorder(2.2)


def draw_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_value: float,
    label: str,
    tick_height: float,
    *,
    fontsize: float | None = None,
) -> None:
    ax.plot([x1, x1, x2, x2], [y_value - tick_height, y_value, y_value, y_value - tick_height],
            color=COLORS["axis"], linewidth=0.8, clip_on=False)
    label_x = (x1 + x2) / 2
    label_y = y_value + tick_height * 0.45
    ax.text(label_x, label_y, label, ha="center", va="bottom", fontsize=fontsize)


def write_stats(path: Path, stats: dict[str, object]) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    if text.endswith("\n"):
        cleaned += "\n"
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    panel: str,
    *,
    bbox_inches: str | None = "tight",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{panel}.png"
    svg_path = output_dir / f"{panel}.svg"
    save_kwargs = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(png_path, dpi=DPI, transparent=True, **save_kwargs)
    fig.savefig(svg_path, format="svg", transparent=True, **save_kwargs)
    plt.close(fig)
    strip_trailing_whitespace(svg_path)

    png_size_mb = png_path.stat().st_size / 1024 / 1024
    svg_size_kb = svg_path.stat().st_size / 1024
    if png_size_mb > 5:
        raise RuntimeError(f"PNG {png_path} is {png_size_mb:.2f} MB, exceeding 5 MB.")
    if svg_size_kb > 500:
        raise RuntimeError(f"SVG {svg_path} is {svg_size_kb:.1f} KB, exceeding 500 KB.")
    print(f"File sizes OK for Figure 7{panel}: PNG {png_size_mb:.2f} MB, SVG {svg_size_kb:.1f} KB")
    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "png_size_mb": png_size_mb,
        "svg_size_kb": svg_size_kb,
    }


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def load_csv(clean_dir: Path, filename: str) -> pd.DataFrame:
    path = clean_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def apply_figure7b_til_trim28_norm_corrections(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Correct source-workbook decimal-place error in Exp61 tumor Trim28 normalization."""
    table = table.copy()
    corrected = 0
    if "sample" not in table.columns or "TIL_Trim28_norm" not in table.columns:
        return table, corrected

    values = pd.to_numeric(table["TIL_Trim28_norm"], errors="coerce")
    for sample_prefix, multiplier in FIGURE7B_TIL_TRIM28_NORM_CORRECTIONS.items():
        mask = table["sample"].astype(str).str.startswith(sample_prefix) & values.notna()
        corrected += int(mask.sum())
        table.loc[mask, "TIL_Trim28_norm"] = values.loc[mask] * multiplier
    return table, corrected


def panel_a(clean_dir: Path, output_dir: Path) -> None:
    table = load_csv(clean_dir, PANEL_TABLES["a"])
    require_columns(
        table,
        ["sample", "NT_TCF7_percent", "Trim28_TCF7_percent", "Trim28_minus_NT"],
        str(PANEL_TABLES["a"]),
    )
    table = table.sort_values(["experiment", "sample"]).copy()
    nt_values = table["NT_TCF7_percent"].astype(float).tolist()
    trim_values = table["Trim28_TCF7_percent"].astype(float).tolist()
    diffs = table["Trim28_minus_NT"].astype(float).tolist()
    wilcoxon_stats = wilcoxon_signed_rank_two_sided(diffs)
    p_value = wilcoxon_stats["p"]
    nt_x_base, trim_x_base = TWO_GROUP_X

    fig, ax = plt.subplots(figsize=TWO_GROUP_FIGSIZE)
    style_axis(ax, [0, 20, 40, 60, 80])
    ax.set_xlim(*TWO_GROUP_XLIM)
    ax.set_ylim(0, 85)
    ax.set_xticks([nt_x_base, trim_x_base], ["NT", "Trim28"])
    significance_label_size = plt.rcParams["font.size"]
    ax.set_xlabel(f"paired n={len(table)}", fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_ylabel("% of tumor CD8 T cells")
    ax.set_title("TCF7+", pad=12)
    draw_boxplot(ax, nt_x_base, nt_values, COLORS["nt"])
    draw_boxplot(ax, trim_x_base, trim_values, COLORS["trim"])

    for _, row in table.iterrows():
        sample = row["sample"]
        nt_x = nt_x_base + stable_jitter(f"{sample}-nt", TWO_GROUP_JITTER)
        trim_x = trim_x_base + stable_jitter(f"{sample}-trim", TWO_GROUP_JITTER)
        ax.plot(
            [nt_x, trim_x],
            [row["NT_TCF7_percent"], row["Trim28_TCF7_percent"]],
            color=COLORS["light"],
            linewidth=0.55,
            zorder=2.5,
        )
        ax.scatter(
            nt_x,
            row["NT_TCF7_percent"],
            s=20,
            facecolor=COLORS["nt"],
            edgecolor=COLORS["nt"],
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            trim_x,
            row["Trim28_TCF7_percent"],
            s=20,
            facecolor=COLORS["trim"],
            edgecolor=COLORS["trim_dark"],
            linewidth=0.5,
            zorder=4,
        )

    bracket_y = 80
    bracket_tick = 2.2
    draw_bracket(
        ax,
        nt_x_base,
        trim_x_base,
        bracket_y,
        format_p(p_value),
        bracket_tick,
        fontsize=significance_label_size,
    )
    fig.subplots_adjust(**TWO_GROUP_SUBPLOTS_ADJUST)

    stats = {
        "panel": "7A",
        "n_pairs": int(len(table)),
        "nt_mean": mean(nt_values),
        "trim28_mean": mean(trim_values),
        "mean_difference_trim28_minus_nt": mean(diffs),
        "nt_sem": sem(nt_values),
        "trim28_sem": sem(trim_values),
        "wilcoxon_signed_rank": {
            "test": "Two-sided paired Wilcoxon signed-rank test comparing matched Trim28 and NT TCF7 percentages.",
            **wilcoxon_stats,
        },
    }
    stats.update(save_figure(fig, output_dir, "a", bbox_inches=None))
    write_stats(output_dir / "a_stats.json", stats)
    print(json.dumps(stats, indent=2))


def panel_b(clean_dir: Path, output_dir: Path) -> None:
    table = load_csv(clean_dir, PANEL_TABLES["b"])
    require_columns(
        table,
        ["sample", "dLN_Trim28_norm", "TIL_Trim28_norm"],
        str(PANEL_TABLES["b"]),
    )
    table, corrected_exp61_tumor_n = apply_figure7b_til_trim28_norm_corrections(table)
    dln = pd.to_numeric(table["dLN_Trim28_norm"], errors="coerce")
    tumor = pd.to_numeric(table["TIL_Trim28_norm"], errors="coerce")
    complete_pairs = table.loc[dln.notna() & tumor.notna()].copy()
    dln_values = dln.dropna().astype(float).tolist()
    tumor_values = tumor.dropna().astype(float).tolist()
    unpaired_stats = mann_whitney_two_sided(dln_values, tumor_values)
    dln_x_base, tumor_x_base = TWO_GROUP_X

    fig, ax = plt.subplots(figsize=TWO_GROUP_FIGSIZE)
    style_axis(ax, [0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlim(*TWO_GROUP_XLIM)
    ax.set_ylim(0, 2.2)
    ax.set_xticks(
        [dln_x_base, tumor_x_base],
        [f"dLN\nn={len(dln_values)}", f"tumor\nn={len(tumor_values)}"],
    )
    ax.set_ylabel("output% / input%")
    ax.set_title("Trim28 gRNA+ T cells", pad=12)
    ax.axhline(1, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=0.8)
    draw_boxplot(ax, dln_x_base, dln_values, COLORS["axis"])
    draw_boxplot(ax, tumor_x_base, tumor_values, COLORS["axis"])

    for _, row in table.iterrows():
        sample = row["sample"]
        if pd.notna(row["dLN_Trim28_norm"]):
            dln_x = dln_x_base + stable_jitter(f"{sample}-dln", TWO_GROUP_JITTER)
            ax.scatter(
                dln_x,
                float(row["dLN_Trim28_norm"]),
                s=20,
                facecolor=COLORS["axis"],
                edgecolor=COLORS["axis"],
                linewidth=0.45,
                alpha=0.88,
                zorder=4,
            )
        if pd.notna(row["TIL_Trim28_norm"]):
            tumor_x = tumor_x_base + stable_jitter(f"{sample}-tumor", TWO_GROUP_JITTER)
            ax.scatter(
                tumor_x,
                float(row["TIL_Trim28_norm"]),
                s=20,
                facecolor=COLORS["axis"],
                edgecolor=COLORS["axis"],
                linewidth=0.45,
                alpha=0.88,
                zorder=4,
            )

    bracket_y = 2.05
    bracket_tick = 0.06
    draw_bracket(
        ax,
        dln_x_base,
        tumor_x_base,
        bracket_y,
        format_p(unpaired_stats["p"]),
        bracket_tick,
    )
    fig.subplots_adjust(**TWO_GROUP_SUBPLOTS_ADJUST)

    stats = {
        "panel": "7B",
        "dln_n": int(len(dln_values)),
        "tumor_n": int(len(tumor_values)),
        "paired_complete_n": int(len(complete_pairs)),
        "dln_mean": mean(dln_values),
        "tumor_mean": mean(tumor_values),
        "data_corrections": {
            "exp61_tumor_trim28_norm_multiplier": FIGURE7B_TIL_TRIM28_NORM_CORRECTIONS["exp61_"],
            "corrected_exp61_tumor_points": corrected_exp61_tumor_n,
            "reason": "The source workbook Exp61 tumor Trim28 normalized values are 10x too small relative to the TIL Trim28 percentage and the matched Exp61 input fraction.",
        },
        "mann_whitney_rank_sum": {
            "test": "Two-sided Mann-Whitney U / Wilcoxon rank-sum test comparing dLN and tumor values as unpaired groups.",
            "dln_n": unpaired_stats["control_n"],
            "tumor_n": unpaired_stats["anti_pd1_n"],
            "dln_mean": unpaired_stats["control_mean"],
            "tumor_mean": unpaired_stats["anti_pd1_mean"],
            "dln_median": unpaired_stats["control_median"],
            "tumor_median": unpaired_stats["anti_pd1_median"],
            "u_dln": unpaired_stats["u_control"],
            "u_tumor": unpaired_stats["u_anti_pd1"],
            "z": unpaired_stats["z"],
            "p": unpaired_stats["p"],
        },
    }
    stats.update(save_figure(fig, output_dir, "b", bbox_inches=None))
    write_stats(output_dir / "b_stats.json", stats)
    print(json.dumps(stats, indent=2))


def draw_pd1_metric_panel(ax: plt.Axes, table: pd.DataFrame, metric: str, y_label: str) -> dict[str, float]:
    control_x_base, pd1_x_base = TWO_GROUP_X
    values = pd.to_numeric(table[metric], errors="coerce")
    metric_table = table.loc[values.notna()].copy()
    metric_table[metric] = values.loc[values.notna()].astype(float)
    control_values = metric_table.loc[
        metric_table["treatment_for_plot"] == "Control", metric
    ].astype(float).tolist()
    pd1_values = metric_table.loc[
        metric_table["treatment_for_plot"] == "anti-PD1", metric
    ].astype(float).tolist()
    metric_stats = mann_whitney_two_sided(control_values, pd1_values)

    style_axis(ax, [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlim(*TWO_GROUP_XLIM)
    ax.set_ylim(0, 2.85)
    ax.set_xticks(
        [control_x_base, pd1_x_base],
        [f"Control\nn={len(control_values)}", f"anti-PD-1\nn={len(pd1_values)}"],
    )
    ax.set_ylabel(y_label)
    ax.set_title("Trim28 gRNA+ T cells", pad=12)
    ax.axhline(1, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=0.8)
    control_color = COLORS["isotype"]
    pd1_color = COLORS["apd1"]
    control_edge_color = blend_color(control_color, "black", 0.25)
    pd1_edge_color = blend_color(pd1_color, "black", 0.25)
    draw_boxplot(ax, control_x_base, control_values, control_color)
    draw_boxplot(ax, pd1_x_base, pd1_values, pd1_color)

    for _, row in metric_table.iterrows():
        is_pd1 = row["treatment_for_plot"] == "anti-PD1"
        x_base = pd1_x_base if is_pd1 else control_x_base
        x_value = x_base + stable_jitter(
            f"{metric}-{row['experiment']}-{row['sample']}-{row['source_type']}",
            TWO_GROUP_JITTER,
        )
        ax.scatter(
            x_value,
            float(row[metric]),
            s=20,
            facecolor=pd1_color if is_pd1 else control_color,
            edgecolor=pd1_edge_color if is_pd1 else control_edge_color,
            linewidth=0.55 if is_pd1 else 0.8,
            zorder=4,
        )
    draw_bracket(ax, control_x_base, pd1_x_base, 2.66, format_p(metric_stats["p"]), 0.07)
    return metric_stats


def panel_pd1_metric(clean_dir: Path, output_dir: Path, panel: str) -> None:
    table = load_csv(clean_dir, PANEL_TABLES[panel])
    require_columns(table, PD1_REQUIRED_COLUMNS, str(PANEL_TABLES[panel]))
    metric, y_label = PD1_METRICS[panel]

    fig, ax = plt.subplots(figsize=TWO_GROUP_FIGSIZE)
    metric_stats = draw_pd1_metric_panel(ax, table, metric, y_label)
    fig.subplots_adjust(**TWO_GROUP_SUBPLOTS_ADJUST)

    stats: dict[str, object] = {
        "panel": f"7{panel.upper()}",
        "metric": metric,
        "metrics": {metric: metric_stats},
    }
    stats.update(save_figure(fig, output_dir, panel, bbox_inches=None))
    write_stats(output_dir / f"{panel}_stats.json", stats)
    print(json.dumps(stats, indent=2))


def panel_c(clean_dir: Path, output_dir: Path) -> None:
    panel_pd1_metric(clean_dir, output_dir, "c")


def panel_d(clean_dir: Path, output_dir: Path) -> None:
    panel_pd1_metric(clean_dir, output_dir, "d")


def build_d_experimental_table(clean_dir: Path, parsed_dir: Path) -> pd.DataFrame:
    included = load_csv(clean_dir, PANEL_TABLES["d_experimental"])
    require_columns(
        included,
        ["experiment", "sample", "source_type", "treatment_for_plot", "TIL_over_dLN_Trim28_ratio"],
        str(PANEL_TABLES["d_experimental"]),
    )
    included = included.loc[
        pd.to_numeric(included["TIL_over_dLN_Trim28_ratio"], errors="coerce").notna()
    ].copy()
    included["sample_key"] = included["sample"].astype(str)

    flow = load_csv(clean_dir, "Fig7C_PD1_enrichment_curated_flow_clean.csv")
    require_columns(
        flow,
        ["experiment", "sample", "dLN_Trim28_percent", "TIL_Trim28_percent"],
        "Fig7C_PD1_enrichment_curated_flow_clean.csv",
    )
    flow["sample_key"] = flow["sample"].astype(str)
    flow_lookup = {
        (row["experiment"], row["sample_key"]): row for _, row in flow.iterrows()
    }

    exp74_path = parsed_dir / "fig7bce_exp74_guide_counts_summary.csv"
    exp74 = pd.read_csv(exp74_path)
    require_columns(
        exp74,
        ["mouse", "dLN_Trim28_fraction", "TIL_Trim28_fraction"],
        str(exp74_path),
    )
    exp74["sample_key"] = "mouse " + exp74["mouse"].astype(str)
    exp74_lookup = {row["sample_key"]: row for _, row in exp74.iterrows()}

    records: list[dict[str, object]] = []
    for _, row in included.iterrows():
        sample_key = row["sample_key"]
        source = row["source_type"]
        if source == "flow cytometry":
            raw = flow_lookup.get((row["experiment"], sample_key))
            if raw is None:
                raise ValueError(f"Missing raw flow fractions for {row['experiment']} / {sample_key}")
            dln_fraction = float(raw["dLN_Trim28_percent"]) / 100.0
            tumor_fraction = float(raw["TIL_Trim28_percent"]) / 100.0
        elif source == "guide counts" and row["experiment"] == "Exp74":
            raw = exp74_lookup.get(sample_key)
            if raw is None:
                raise ValueError(f"Missing raw Exp74 guide-count fractions for {sample_key}")
            dln_fraction = float(raw["dLN_Trim28_fraction"])
            tumor_fraction = float(raw["TIL_Trim28_fraction"])
        else:
            raise ValueError(f"Unsupported source row for d_experimental: {source}")

        sample_id = f"{row['experiment']}::{sample_key}::{source}"
        for tissue, fraction in (("dLN", dln_fraction), ("Tumor", tumor_fraction)):
            records.append(
                {
                    "sample_id": sample_id,
                    "experiment": row["experiment"],
                    "sample": sample_key,
                    "source_type": source,
                    "treatment_for_plot": row["treatment_for_plot"],
                    "tissue": tissue,
                    "Trim28_fraction": fraction,
                }
            )
    return pd.DataFrame.from_records(records)


def panel_d_experimental(clean_dir: Path, output_dir: Path, parsed_dir: Path) -> None:
    table = build_d_experimental_table(clean_dir, parsed_dir)
    table = table.loc[table["treatment_for_plot"] == "anti-PD1"].copy()
    if table.empty:
        raise ValueError("No anti-PD-1 rows available for d_experimental.")
    tissue_order = ["dLN", "Tumor"]
    x_by_tissue = {"dLN": TWO_GROUP_X[0], "Tumor": TWO_GROUP_X[1]}
    point_color = COLORS["apd1"]

    paired = table.pivot(index="sample_id", columns="tissue", values="Trim28_fraction")
    diffs = (paired["Tumor"] - paired["dLN"]).astype(float).tolist()
    p_value = exact_paired_signflip_p(diffs)

    fig, ax = plt.subplots(figsize=TWO_GROUP_FIGSIZE)
    style_axis(ax, [0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(*TWO_GROUP_XLIM)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([x_by_tissue[tissue] for tissue in tissue_order], ["dLN", "tumor"])
    ax.set_ylabel("Trim28 fraction")

    for tissue in tissue_order:
        values = table.loc[table["tissue"] == tissue, "Trim28_fraction"].astype(float).tolist()
        draw_boxplot(ax, x_by_tissue[tissue], values, COLORS["muted"])

    sample_rows = table.pivot_table(
        index=["sample_id", "treatment_for_plot"],
        columns="tissue",
        values="Trim28_fraction",
        aggfunc="first",
    ).reset_index()
    for _, row in sample_rows.iterrows():
        dln_x = x_by_tissue["dLN"] + stable_jitter(f"{row['sample_id']}-dln", TWO_GROUP_JITTER)
        tumor_x = x_by_tissue["Tumor"] + stable_jitter(f"{row['sample_id']}-tumor", TWO_GROUP_JITTER)
        ax.plot(
            [dln_x, tumor_x],
            [float(row["dLN"]), float(row["Tumor"])],
            color=blend_color(point_color, "white", 0.35),
            linewidth=0.55,
            alpha=0.78,
            zorder=2.5,
        )
        for x_value, tissue in ((dln_x, "dLN"), (tumor_x, "Tumor")):
            ax.scatter(
                x_value,
                float(row[tissue]),
                s=20,
                facecolor=point_color,
                edgecolor=blend_color(point_color, "black", 0.25),
                linewidth=0.55,
                alpha=0.9,
                zorder=4,
            )

    n_pairs = int(table["sample_id"].nunique())
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=point_color,
            markeredgecolor=blend_color(point_color, "black", 0.25),
            markersize=5,
            label=f"anti-PD-1 n={n_pairs}",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8, handletextpad=0.4)
    fig.subplots_adjust(**TWO_GROUP_SUBPLOTS_ADJUST)

    tissue_summary = []
    for tissue in tissue_order:
        values = table.loc[table["tissue"] == tissue, "Trim28_fraction"].astype(float).tolist()
        tissue_summary.append(
            {
                "tissue": tissue,
                "n": len(values),
                "mean": mean(values),
                "median": median(values),
                "sem": sem(values),
                "min": min(values),
                "max": max(values),
            }
        )

    stats = {
        "panel": "7D_experimental",
        "treatment": "anti-PD1",
        "n_pairs": int(len(paired)),
        "tissue_summary": tissue_summary,
        "paired_tumor_minus_dln": {
            "mean": mean(diffs),
            "median": median(diffs),
            "paired_signflip_p": p_value,
        },
    }
    stats.update(save_figure(fig, output_dir, "d_experimental", bbox_inches=None))
    write_stats(output_dir / "d_experimental_stats.json", stats)
    print(json.dumps(stats, indent=2))


def group_counts(rows: pd.DataFrame) -> dict[str, int]:
    return {
        group: int(rows.loc[rows["group"] == group, "mouse_id"].nunique())
        for group in ["NT", "Trim28"]
    }


def group_legend_handles(counts: dict[str, int]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS["nt"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"NT n={counts['NT']}",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["trim"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"Trim28 n={counts['Trim28']}",
        ),
    ]


def logrank_two_group(rows: pd.DataFrame, focal_group: str = "Trim28") -> dict[str, float | None]:
    groups = sorted(rows["group"].astype(str).unique())
    if len(groups) != 2 or focal_group not in groups:
        return {"chi_square": None, "p": None, "observed_minus_expected": None, "variance": None}

    event_rows = rows.loc[rows["event"].astype(int) == 1].copy()
    event_times = sorted(event_rows["time_day"].astype(float).unique())
    oe = 0.0
    variance = 0.0
    for time in event_times:
        at_risk = rows.loc[rows["time_day"].astype(float) >= time]
        events = rows.loc[
            (rows["time_day"].astype(float) == time) & (rows["event"].astype(int) == 1)
        ]
        n_risk = len(at_risk)
        n_events = len(events)
        if n_risk <= 1 or n_events == 0:
            continue
        n_focal = int((at_risk["group"] == focal_group).sum())
        n_other = n_risk - n_focal
        d_focal = int((events["group"] == focal_group).sum())
        expected = n_events * n_focal / n_risk
        var_increment = (
            n_focal * n_other * n_events * (n_risk - n_events)
            / (n_risk * n_risk * (n_risk - 1))
        )
        oe += d_focal - expected
        variance += var_increment
    if variance <= 0:
        return {"chi_square": None, "p": None, "observed_minus_expected": oe, "variance": variance}
    chi_square = (oe * oe) / variance
    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    return {
        "chi_square": chi_square,
        "p": p_value,
        "observed_minus_expected": oe,
        "variance": variance,
    }


def km_points(rows: pd.DataFrame, group: str, x_max: float) -> list[tuple[float, float]]:
    group_rows = rows.loc[rows["group"] == group].copy()
    survival = 1.0
    n_at_risk = len(group_rows)
    points = [(0.0, 100.0)]
    for time in sorted(group_rows["time_day"].astype(float).unique()):
        at_time = group_rows.loc[group_rows["time_day"].astype(float) == time]
        n_events = int(at_time["event"].astype(int).sum())
        if n_events:
            points.append((time, survival * 100))
            survival *= 1 - n_events / n_at_risk
            points.append((time, survival * 100))
        n_at_risk -= len(at_time)
    points.append((x_max, survival * 100))
    return points


def survival_at(rows: pd.DataFrame, group: str, time_day: float) -> float:
    group_rows = rows.loc[rows["group"] == group].copy()
    survival = 1.0
    n_at_risk = len(group_rows)
    for time in sorted(group_rows["time_day"].astype(float).unique()):
        at_time = group_rows.loc[group_rows["time_day"].astype(float) == time]
        n_events = int(at_time["event"].astype(int).sum())
        if n_events:
            survival *= 1 - n_events / n_at_risk
        n_at_risk -= len(at_time)
        if time >= time_day:
            break
    return survival * 100


def draw_growth(
    ax: plt.Axes,
    rows: pd.DataFrame,
    title: str,
    x_min: float,
    x_max: float,
    x_ticks: list[float],
    show_ylabel: bool,
    show_xlabel: bool,
    title_color: str = COLORS["axis"],
) -> None:
    style_axis(ax, [0, 1000, 2000, 3000])
    ax.set_title(title, pad=8, color=title_color, fontsize=GROWTH_TREATMENT_TITLE_SIZE)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 3600)
    ax.set_xticks(x_ticks, [format_axis_value(x) for x in x_ticks])
    ax.set_xlabel("Day after tumor challenge" if show_xlabel else "")
    ax.set_ylabel("Tumor volume (mm3)" if show_ylabel else "")

    for (mouse_id, group), mouse_rows in rows.groupby(["mouse_id", "group"]):
        mouse_rows = mouse_rows.sort_values("day")
        color = GROWTH_NT_BACKGROUND if group == "NT" else GROWTH_TRIM28_BACKGROUND
        ax.plot(
            mouse_rows["day"].astype(float),
            mouse_rows["tumor_volume_mm3"].astype(float),
            color=color,
            linewidth=GROWTH_BACKGROUND_LINE_WIDTH,
            alpha=GROWTH_BACKGROUND_LINE_ALPHA,
            zorder=1,
        )

    for group, color in [("NT", COLORS["nt"]), ("Trim28", COLORS["trim"])]:
        group_rows = rows.loc[rows["group"] == group].copy()
        if group_rows.empty:
            continue
        grouped = group_rows.groupby("day")["tumor_volume_mm3"]
        days = [float(day) for day in grouped.mean().index]
        means = grouped.mean().astype(float).tolist()
        errors = [sem(vals.astype(float).tolist()) for _, vals in grouped]
        ax.plot(days, means, color=color, linewidth=1.4, zorder=4)
        ax.errorbar(
            days,
            means,
            yerr=errors,
            fmt="o",
            color=color,
            markersize=2.8,
            linewidth=0.8,
            capsize=0,
            zorder=5,
        )

def draw_survival(
    ax: plt.Axes,
    rows: pd.DataFrame,
    p_value: float | None,
    x_max: float,
    x_ticks: list[float],
    show_ylabel: bool,
) -> None:
    style_axis(ax, [0, 50, 100])
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_ticks, [format_axis_value(x) for x in x_ticks])
    ax.set_xlabel("Day after tumor challenge")
    ax.set_ylabel("Survival (%)" if show_ylabel else "")
    for group, color in [("NT", COLORS["nt"]), ("Trim28", COLORS["trim"])]:
        points = km_points(rows, group, x_max)
        xs, ys = zip(*points)
        ax.step(xs, ys, where="post", color=color, linewidth=1.4)
        censored = rows.loc[(rows["group"] == group) & (rows["event"].astype(int) == 0)]
        for _, row in censored.iterrows():
            time_day = float(row["time_day"])
            ax.plot(
                [time_day, time_day],
                [survival_at(rows, group, time_day) - 3, survival_at(rows, group, time_day) + 3],
                color=color,
                linewidth=0.8,
            )
    ax.text(0.04, 0.08, f"Log-rank {format_logrank_p(p_value)}", transform=ax.transAxes,
            ha="left", va="bottom")


def summarize_survival(rows: pd.DataFrame, stats: dict[str, float | None]) -> dict[str, object]:
    counts = group_counts(rows)
    return {
        "n": counts,
        "events": {
            group: int(((rows["group"] == group) & (rows["event"].astype(int) == 1)).sum())
            for group in ["NT", "Trim28"]
        },
        "censored": {
            group: int(((rows["group"] == group) & (rows["event"].astype(int) == 0)).sum())
            for group in ["NT", "Trim28"]
        },
        "logrank_chi_square": stats["chi_square"],
        "logrank_p": stats["p"],
    }


def panel_e(clean_dir: Path, output_dir: Path) -> None:
    exp71_growth = load_csv(clean_dir, PANEL_TABLES["e"][0])
    exp71_survival = load_csv(clean_dir, PANEL_TABLES["e"][1])
    exp67_growth = load_csv(clean_dir, PANEL_TABLES["e"][2])
    exp67_survival = load_csv(clean_dir, PANEL_TABLES["e"][3])

    stats71 = logrank_two_group(exp71_survival)
    stats67 = logrank_two_group(exp67_survival)

    with plt.rc_context(GROWTH_SURVIVAL_FONT_PARAMS):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14.2, 10.4),
            gridspec_kw={"height_ratios": [1.15, 0.85]},
        )
        draw_growth(
            axes[0, 0],
            exp71_growth,
            "Isotype",
            7,
            23,
            [7, 13, 19, 23],
            True,
            False,
            COLORS["isotype"],
        )
        draw_growth(
            axes[0, 1],
            exp67_growth,
            "anti-PD-1",
            7,
            33,
            [7, 14, 21, 28, 33],
            False,
            False,
            COLORS["apd1"],
        )
        draw_survival(axes[1, 0], exp71_survival, stats71["p"], 23, [0, 7, 14, 21], True)
        draw_survival(axes[1, 1], exp67_survival, stats67["p"], 40, [0, 10, 20, 30, 40], False)

        axes[0, 0].legend(
            handles=group_legend_handles(group_counts(exp71_survival)),
            loc="upper left",
            frameon=False,
            borderaxespad=0.3,
            handlelength=1.6,
        )
        axes[0, 1].legend(
            handles=group_legend_handles(group_counts(exp67_survival)),
            loc="upper right",
            frameon=False,
            borderaxespad=0.3,
            handlelength=1.6,
        )

        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.20, top=0.80, wspace=0.18, hspace=0.28)

    stats = {
        "panel": "7E",
        "exp71_no_pd1": {
            "growth_rows": int(len(exp71_growth)),
            "survival": summarize_survival(exp71_survival, stats71),
        },
        "exp67_pd1": {
            "growth_rows": int(len(exp67_growth)),
            "survival": summarize_survival(exp67_survival, stats67),
        },
        "survival_rules": {
            "exp71": "Trim28 75% mixture excluded; blank endpoint reasons treated as censored in the clean table.",
            "exp67": "All recorded end days treated as events in the clean table.",
            "test": "Two-sided log-rank/Mantel-Cox test using chi-square distribution with 1 degree of freedom.",
        },
    }
    stats.update(save_figure(fig, output_dir, "e"))
    write_stats(output_dir / "e_stats.json", stats)
    print(json.dumps(stats, indent=2))


def panel_f(clean_dir: Path, output_dir: Path) -> None:
    table = load_csv(clean_dir, PANEL_TABLES["f"][0])
    pairs = load_csv(clean_dir, PANEL_TABLES["f"][1])
    require_columns(
        table,
        ["experiment", "mouse_key", "site", "Trim28_fraction"],
        str(PANEL_TABLES["f"][0]),
    )
    require_columns(
        pairs,
        [
            "experiment",
            "mouse_key",
            "peritumor_Trim28_fraction",
            "distant_Trim28_fraction",
        ],
        str(PANEL_TABLES["f"][1]),
    )
    excluded_experiments = set(SKIN_EXCLUDED_EXPERIMENTS)
    all_paired_mouse_keys = set(pairs["mouse_key"].astype(str))
    excluded_table = table.loc[
        table["experiment"].astype(str).isin(excluded_experiments)
        & table["mouse_key"].astype(str).isin(all_paired_mouse_keys)
    ].copy()
    excluded_pairs = pairs.loc[
        pairs["experiment"].astype(str).isin(excluded_experiments)
    ].copy()
    table = table.loc[~table["experiment"].astype(str).isin(excluded_experiments)].copy()
    pairs = pairs.loc[~pairs["experiment"].astype(str).isin(excluded_experiments)].copy()
    paired_mouse_keys = set(pairs["mouse_key"].astype(str))
    table = table.loc[table["mouse_key"].astype(str).isin(paired_mouse_keys)].copy()
    site_order = ["TIL", "Peritumor skin", "Distant skin"]
    site_colors = {site: COLORS["axis"] for site in site_order}

    def input_fraction(experiment: object) -> float:
        experiment_key = str(experiment)
        if experiment_key not in SKIN_INPUT_TRIM28_FRACTIONS:
            raise ValueError(
                f"No Trim28 input fraction configured for Figure 7f experiment {experiment_key!r}."
            )
        return SKIN_INPUT_TRIM28_FRACTIONS[experiment_key]

    table["input_Trim28_fraction"] = table["experiment"].map(input_fraction)
    table["Trim28_output_input"] = (
        pd.to_numeric(table["Trim28_fraction"], errors="coerce") / table["input_Trim28_fraction"]
    )
    pairs = pairs.copy()
    pairs["input_Trim28_fraction"] = pairs["experiment"].map(input_fraction)
    pairs["peritumor_Trim28_output_input"] = (
        pd.to_numeric(pairs["peritumor_Trim28_fraction"], errors="coerce")
        / pairs["input_Trim28_fraction"]
    )
    pairs["distant_Trim28_output_input"] = (
        pd.to_numeric(pairs["distant_Trim28_fraction"], errors="coerce")
        / pairs["input_Trim28_fraction"]
    )
    pairs["distant_minus_peritumor_output_input"] = (
        pairs["distant_Trim28_output_input"] - pairs["peritumor_Trim28_output_input"]
    )
    diffs = pairs["distant_minus_peritumor_output_input"].astype(float).tolist()
    p_value = exact_paired_signflip_p(diffs)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    style_axis(ax, [0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("output% / input%")
    ax.set_title("Trim28 gRNA+ T cells", pad=12)
    ax.axhline(1, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=0.8)

    x_by_site = {site: idx for idx, site in enumerate(site_order)}
    for _, pair in pairs.iterrows():
        peritumor_x = x_by_site["Peritumor skin"] + stable_jitter(
            f"{pair['mouse_key']}-Peritumor skin", 0.18
        )
        distant_x = x_by_site["Distant skin"] + stable_jitter(
            f"{pair['mouse_key']}-Distant skin", 0.18
        )
        ax.plot(
            [peritumor_x, distant_x],
            [
                pair["peritumor_Trim28_output_input"],
                pair["distant_Trim28_output_input"],
            ],
            color=COLORS["light"],
            linewidth=0.55,
            zorder=2.5,
        )

    site_summary = []
    site_counts = {}
    for site in site_order:
        site_table = table.loc[table["site"] == site]
        plot_values = site_table["Trim28_output_input"].astype(float).tolist()
        site_counts[site] = len(plot_values)
        site_summary.append(
            {
                "site": site,
                "n": len(plot_values),
                "mean_output_input": mean(plot_values),
                "median_output_input": median(plot_values),
                "sem_output_input": sem(plot_values),
                "min_output_input": min(plot_values),
                "max_output_input": max(plot_values),
            }
        )
        draw_boxplot(ax, x_by_site[site], plot_values, site_colors[site])
        for _, row in site_table.iterrows():
            x_value = x_by_site[site] + stable_jitter(f"{row['mouse_key']}-{site}", 0.18)
            ax.scatter(
                x_value,
                float(row["Trim28_output_input"]),
                s=20,
                facecolor=site_colors[site],
                edgecolor=site_colors[site],
                linewidth=0.45,
                alpha=0.88,
                zorder=4,
            )

    ax.set_xticks(
        range(len(site_order)),
        [
            f"{site}\npaired n={len(pairs)}"
            if site in {"Peritumor skin", "Distant skin"}
            else f"{site}\nn={site_counts[site]}"
            for site in site_order
        ],
    )

    bracket_y = 1.74
    bracket_tick = 0.06
    draw_bracket(
        ax,
        x_by_site["Peritumor skin"],
        x_by_site["Distant skin"],
        bracket_y,
        format_paired_p(p_value, len(pairs)),
        bracket_tick,
    )

    stats = {
        "panel": "7F",
        "normalization": {
            "metric": "Output Trim28 percentage divided by experiment-specific input Trim28 percentage.",
            "input_Trim28_fractions": SKIN_INPUT_TRIM28_FRACTIONS,
            "input_Trim28_fraction_notes": SKIN_INPUT_TRIM28_FRACTION_NOTES,
            "excluded_experiments": SKIN_EXCLUDED_EXPERIMENTS,
            "excluded_experiment_counts": {
                experiment: {
                    "plotted_site_rows": int(
                        (excluded_table["experiment"].astype(str) == experiment).sum()
                    ),
                    "paired_peritumor_distant_pairs": int(
                        (excluded_pairs["experiment"].astype(str) == experiment).sum()
                    ),
                }
                for experiment in sorted(excluded_experiments)
            },
        },
        "site_summary": site_summary,
        "paired_peritumor_vs_distant": {
            "n": int(len(pairs)),
            "mean_distant_minus_peritumor_output_input": mean(diffs),
            "median_distant_minus_peritumor_output_input": median(diffs),
            "positive_pairs": int(sum(1 for value in diffs if value > 0)),
            "negative_pairs": int(sum(1 for value in diffs if value < 0)),
            "paired_signflip_p": p_value,
        },
    }
    stats.update(save_figure(fig, output_dir, "f"))
    write_stats(output_dir / "f_stats.json", stats)
    print(json.dumps(stats, indent=2))


def main() -> None:
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    parsed_dir = Path(args.parsed_dir)
    output_dir = Path(args.output_dir)

    if args.panel == "d_experimental":
        panel_d_experimental(clean_dir, output_dir, parsed_dir)
        return

    panel_functions = {
        "a": panel_a,
        "b": panel_b,
        "c": panel_c,
        "d": panel_d,
        "e": panel_e,
        "f": panel_f,
    }
    panel_functions[args.panel](clean_dir, output_dir)


if __name__ == "__main__":
    main()
