#!/usr/bin/env python
"""
Generate Figure 1a/b scatter plots for the CP1 and CP3 in vivo Tcf7 screens.

Each point is a guide. The x-axis is Tcf7- guide abundance and the y-axis is
Tcf7+ guide abundance. A control-guide negative-binomial fit with a +1
pseudocount in the modeled counts is used to define the null relationship, and
labeled targeting guides highlight outliers.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom
from statsmodels.stats.multitest import multipletests


NON_GENE_TEXT_SCALE = 1.5


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["axes.titlesize"] = 18 * NON_GENE_TEXT_SCALE
plt.rcParams["axes.labelsize"] = 18 * NON_GENE_TEXT_SCALE
plt.rcParams["xtick.labelsize"] = 14 * NON_GENE_TEXT_SCALE
plt.rcParams["ytick.labelsize"] = 14 * NON_GENE_TEXT_SCALE
plt.rcParams["figure.figsize"] = (7, 7)
plt.rcParams["axes.grid"] = False


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "manuscript" / "supplementary_tables"
CONTROL_PREFIXES = ("NO_SITE", "ONE_INTERGENIC")
CONTROL_COLOR = "#9aa0a6"
KO_COLOR = "#d62728"
PSEUDOCOUNT = 1.0

SCREEN_CONFIG = {
    "cp1": {
        "counts_path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "5_23_22_CP1_B16"
        / "poolq"
        / "counts.txt",
        "x_col": "TIL_GFP_Positive",
        "y_col": "TIL_GFP_Negative",
        "panel_letter": "a",
        "table_suffix": "1_4",
    },
    "cp3": {
        "counts_path": REPO_ROOT
        / "raw_data"
        / "CRISPR_sort_screens"
        / "7_13_22_CP3_B16"
        / "poolq"
        / "counts.txt",
        "x_col": "tcf7 pos",
        "y_col": "tcf7 neg",
        "panel_letter": "b",
        "table_suffix": "1_5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", choices=sorted(SCREEN_CONFIG), required=True)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "manuscript" / "plots" / "Figure1"),
    )
    parser.add_argument("--panel-letter")
    parser.add_argument("--table-suffix")
    parser.add_argument("--guide-stats-output")
    parser.add_argument("--fit-summary-output")
    parser.add_argument("--fit-intercept", action="store_true")
    return parser.parse_args()


def classify_guide(guide_name: str) -> str:
    return "Control" if guide_name.startswith(CONTROL_PREFIXES) else "KO"


def display_label(guide_name: str) -> str:
    if guide_name.startswith("NO_SITE_"):
        return guide_name.replace("NO_SITE_", "NO_SITE-")
    if guide_name.startswith("ONE_INTERGENIC_SITE_"):
        return guide_name.replace("ONE_INTERGENIC_SITE_", "INTERGENIC-")
    return guide_name


def extract_gene(guide_name: str) -> str:
    return re.sub(r"-\d+$", "", guide_name)


def nice_axis_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = int(np.floor(np.log10(value)))
    base = 10 ** exponent
    scaled = value / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * base)


def fit_control_band(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    fit_intercept: bool = False,
) -> dict[str, object]:
    control = df[df["guide_class"] == "Control"].copy()
    x = control[x_col].astype(float).to_numpy() + PSEUDOCOUNT
    y = control[y_col].astype(float).to_numpy() + PSEUDOCOUNT

    def neg_ll_nb(params: np.ndarray, obs: np.ndarray, x_conc: np.ndarray) -> float:
        if fit_intercept:
            r, intercept, slope = params
            mu = intercept + x_conc * slope
        else:
            r, slope = params
            intercept = 0.0
            mu = x_conc * slope
        p = r / (r + mu + 1e-10)
        return -np.sum(nbinom.logpmf(obs, r, p))

    r_init = 1.0
    slope_init = float(np.mean(y) / np.mean(x)) if np.mean(x) > 0 else 1.0
    if fit_intercept:
        intercept_init = max(float(np.mean(y) - slope_init * np.mean(x)), 0.1)
        fit = minimize(
            neg_ll_nb,
            [r_init, intercept_init, slope_init],
            args=(y, x),
            bounds=[(0.001, None), (0.0, None), (0.001, None)],
            method="L-BFGS-B",
        )
        r_star, intercept, slope = fit.x
    else:
        fit = minimize(
            neg_ll_nb,
            [r_init, slope_init],
            args=(y, x),
            bounds=[(0.001, None), (0.001, None)],
            method="L-BFGS-B",
        )
        r_star, slope = fit.x
        intercept = 0.0
    alpha_nb = 1 / r_star
    log_likelihood = -float(fit.fun)
    parameter_count = 3 if fit_intercept else 2
    aic = 2 * parameter_count + 2 * float(fit.fun)

    # Sample the fitted curve densely in log space so the line and predictive
    # band remain smooth when displayed on log-log axes with a +1 pseudocount.
    x_line = np.geomspace(1.0, float(df[x_col].max()) * 1.1 + PSEUDOCOUNT, 2000) - PSEUDOCOUNT
    x_line_model = x_line + PSEUDOCOUNT
    mu_line_model = intercept + slope * x_line_model
    p_line = r_star / (r_star + mu_line_model + 1e-10)
    y_line = np.clip(mu_line_model - PSEUDOCOUNT, 0.0, None)
    y_lower = np.clip(nbinom.ppf(0.025, r_star, p_line) - PSEUDOCOUNT, 0.0, None)
    y_upper = np.clip(nbinom.ppf(0.975, r_star, p_line) - PSEUDOCOUNT, 0.0, None)

    x_all = df[x_col].astype(float).to_numpy()
    y_all = df[y_col].astype(float).to_numpy()
    x_all_model = x_all + PSEUDOCOUNT
    y_all_model = y_all + PSEUDOCOUNT
    mu_all_model = intercept + x_all_model * slope
    stdev_all = np.sqrt(mu_all_model + alpha_nb * (mu_all_model ** 2))
    z_all = (y_all_model - mu_all_model) / (stdev_all + 1e-10)

    p_param_all = r_star / (r_star + mu_all_model + 1e-10)
    lower_tail = nbinom.cdf(y_all_model, r_star, p_param_all)
    upper_tail = nbinom.sf(y_all_model - 1, r_star, p_param_all)
    pvals = np.clip(2 * np.minimum(lower_tail, upper_tail), 0, 1)
    _, bh_adj, _, _ = multipletests(pvals, method="fdr_bh")

    results_df = df.copy()
    results_df["expected_mean_tcf7_positive"] = np.clip(
        mu_all_model - PSEUDOCOUNT,
        0.0,
        None,
    )
    results_df["z_score"] = z_all
    results_df["pvalue"] = pvals
    results_df["bh_adj_pvalue"] = bh_adj
    results_df["significance"] = np.where(
        results_df["bh_adj_pvalue"] < 0.05,
        np.where(
            results_df["guide_class"] == "KO",
            "BH-significant KO",
            "BH-significant control",
        ),
        np.where(
            results_df["pvalue"] < 0.05,
            np.where(
                results_df["guide_class"] == "KO",
                "Nominal p<0.05 KO",
                "Nominal p<0.05 control",
            ),
            "Not significant",
        ),
    )
    results_df = results_df.sort_values(
        ["z_score", "pvalue"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "x_line": x_line,
        "y_line": y_line,
        "y_lower": np.clip(y_lower, 0, None),
        "y_upper": y_upper,
        "intercept": intercept,
        "slope": slope,
        "r": r_star,
        "alpha_nb": alpha_nb,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "fit_success": bool(fit.success),
        "fit_message": str(fit.message),
        "fit_intercept": fit_intercept,
        "results_df": results_df,
    }


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def make_plot(
    screen: str,
    output_dir: Path,
    panel_letter: str,
    table_suffix: str | None,
    fit_intercept: bool,
    guide_stats_output: Path | None,
    fit_summary_output: Path | None,
) -> None:
    config = SCREEN_CONFIG[screen]
    counts = pd.read_csv(config["counts_path"], sep="\t").copy()
    counts = counts.rename(
        columns={
            "Row Barcode IDs": "guide",
            config["x_col"]: "tcf7_positive_count",
            config["y_col"]: "tcf7_negative_count",
        }
    )
    counts["guide_class"] = counts["guide"].astype(str).apply(classify_guide)
    counts["gene"] = counts["guide"].astype(str).apply(extract_gene)

    fit = fit_control_band(
        counts,
        x_col="tcf7_negative_count",
        y_col="tcf7_positive_count",
        fit_intercept=fit_intercept,
    )
    results_df = fit["results_df"].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    if table_suffix is not None:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    control_df = results_df[results_df["guide_class"] == "Control"]
    ko_df = results_df[results_df["guide_class"] == "KO"]
    sig_ko_df = ko_df[ko_df["pvalue"] < 0.05].copy()

    ax.scatter(
        control_df["tcf7_negative_count"] + PSEUDOCOUNT,
        control_df["tcf7_positive_count"] + PSEUDOCOUNT,
        color=CONTROL_COLOR,
        s=32,
        alpha=0.8,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        ko_df["tcf7_negative_count"] + PSEUDOCOUNT,
        ko_df["tcf7_positive_count"] + PSEUDOCOUNT,
        color=KO_COLOR,
        s=32,
        alpha=0.8,
        rasterized=True,
        zorder=2,
    )

    ax.plot(
        fit["x_line"] + PSEUDOCOUNT,
        fit["y_line"] + PSEUDOCOUNT,
        color=CONTROL_COLOR,
        linewidth=2,
        zorder=1,
    )
    ax.fill_between(
        fit["x_line"] + PSEUDOCOUNT,
        fit["y_lower"] + PSEUDOCOUNT,
        fit["y_upper"] + PSEUDOCOUNT,
        color=CONTROL_COLOR,
        alpha=0.22,
        zorder=0,
    )

    for _, row in sig_ko_df.iterrows():
        ax.text(
            row["tcf7_negative_count"] + PSEUDOCOUNT,
            row["tcf7_positive_count"] + PSEUDOCOUNT,
            display_label(row["guide"]),
            color=KO_COLOR,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    x_max = nice_axis_upper(
        max(
            float(results_df["tcf7_negative_count"].max()),
            float(np.max(fit["x_line"])),
        )
    )
    y_max = nice_axis_upper(
        max(
            float(results_df["tcf7_positive_count"].max()),
            float(np.max(fit["y_upper"])),
        )
    )

    ax.set_xlabel("Tcf7- guide abundance (+1)")
    ax.set_ylabel("Tcf7+ guide abundance (+1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(PSEUDOCOUNT, x_max + PSEUDOCOUNT)
    ax.set_ylim(PSEUDOCOUNT, y_max + PSEUDOCOUNT)

    plt.tight_layout()

    png_path = output_dir / f"{panel_letter}.png"
    svg_path = output_dir / f"{panel_letter}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    validate_file_sizes(png_path, svg_path)

    table_df = results_df[
        [
            "guide",
            "tcf7_negative_count",
            "tcf7_positive_count",
            "expected_mean_tcf7_positive",
            "z_score",
            "pvalue",
            "bh_adj_pvalue",
        ]
    ].copy()
    table_path: Path | None = guide_stats_output
    if table_path is None and table_suffix is not None:
        table_path = TABLE_DIR / f"{table_suffix}.tsv"
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(table_path, sep="\t", index=False, float_format="%.6e")
        print(f"Saved guide statistics to {table_path}")

    if fit_summary_output is not None:
        fit_summary_output.parent.mkdir(parents=True, exist_ok=True)
        fit_summary_df = pd.DataFrame(
            [
                {
                    "screen": screen,
                    "panel_letter": panel_letter,
                    "fit_intercept": fit["fit_intercept"],
                    "intercept": fit["intercept"],
                    "slope": fit["slope"],
                    "r": fit["r"],
                    "alpha_nb": fit["alpha_nb"],
                    "model_pseudocount": PSEUDOCOUNT,
                    "control_log_likelihood": fit["log_likelihood"],
                    "aic": fit["aic"],
                    "fit_success": fit["fit_success"],
                    "fit_message": fit["fit_message"],
                }
            ]
        )
        fit_summary_df.to_csv(
            fit_summary_output,
            sep="\t",
            index=False,
            float_format="%.6e",
        )
        print(f"Saved fit summary to {fit_summary_output}")

    print(f"\nTop labeled KO guides for {screen.upper()}:")
    print(
        sig_ko_df[
            [
                "guide",
                "tcf7_negative_count",
                "tcf7_positive_count",
                "z_score",
                "pvalue",
                "bh_adj_pvalue",
            ]
        ]
        .sort_values(["pvalue", "z_score"], kind="mergesort")
        .head(20)
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    config = SCREEN_CONFIG[args.screen]
    panel_letter = args.panel_letter or config["panel_letter"]
    if args.table_suffix is not None:
        table_suffix: str | None = args.table_suffix
    elif args.guide_stats_output:
        table_suffix = None
    else:
        table_suffix = config["table_suffix"]
    make_plot(
        screen=args.screen,
        output_dir=Path(args.output_dir),
        panel_letter=panel_letter,
        table_suffix=table_suffix,
        fit_intercept=args.fit_intercept,
        guide_stats_output=Path(args.guide_stats_output) if args.guide_stats_output else None,
        fit_summary_output=Path(args.fit_summary_output) if args.fit_summary_output else None,
    )


if __name__ == "__main__":
    main()
