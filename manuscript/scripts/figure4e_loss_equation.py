#!/usr/bin/env python
"""
Render the loss equation for Figure 4e as standalone PNG and PDF assets.
"""

import os
import sys

import matplotlib.pyplot as plt

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "manuscript", "plots", "Figure4")


def render_equation() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_text_as_figure(
        text: str,
        filename: str,
        figsize: tuple[float, float] = (8, 3),
        fontsize: int = 26,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            text,
            fontsize=fontsize,
            ha="center",
            va="center",
            color="black",
            transform=ax.transAxes,
        )
        pdf_path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight", dpi=300, transparent=True)
        png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, transparent=True)
        plt.close(fig)

    # Individual PDFs for each component
    save_text_as_figure(
        r"$\mathrm{loss}(\theta) = L_{\mathrm{clone\_size}}(\theta) + L_{\mathrm{prog\_frac}}(\theta)$",
        "e_equation_total",
        figsize=(6.5, 2.2),
        fontsize=30,
    )
    save_text_as_figure(
        r"$L_{\mathrm{clone\_size}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{F}_{\mathrm{real}}(x_i) - \hat{F}_{\mathrm{sim}}(x_i; \theta)\right)^2$",
        "e_equation_ecdf",
        figsize=(8.5, 2.6),
        fontsize=26,
    )
    save_text_as_figure(
        r"$L_{\mathrm{prog\_frac}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{g}_{\mathrm{real}}(x_i) - \hat{g}_{\mathrm{sim}}(x_i; \theta)\right)^2$",
        "e_equation_iso",
        figsize=(8.5, 2.6),
        fontsize=26,
    )
    save_text_as_figure(
        r"$\theta = \left(p_{\mathrm{sym}},\, p_{\mathrm{asym}},\, p_{\mathrm{dif}}\right)$",
        "e_equation_theta",
        figsize=(6.0, 2.0),
        fontsize=28,
    )


def main() -> None:
    # Add repo root to sys.path in case this script is called from Snakemake.
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)

    render_equation()


if __name__ == "__main__":
    main()
