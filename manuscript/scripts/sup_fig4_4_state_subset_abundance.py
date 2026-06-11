#!/usr/bin/env python
"""
Generate deprecated internal Sup_Figure_4_4 state-subset abundance panels.

Panels reuse the Figure 4a/4b negative-binomial framework after subsetting the
gene10 dataset to either the progenitor side (Progenitor + Progenitor Cycling)
or differentiated cells (all remaining cell states).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import pandas as pd

sys.path.append("../manuscript/scripts")
from guide_abundance_correlation_analysis import perform_analysis, convert_long_to_short  # noqa: E402

sys.path.append("../scripts/utils")
from figure_format import get_adata  # noqa: E402


PROGENITOR_SIDE = {"Progenitor", "Progenitor Cycling"}
GUIDES_TO_REMOVE = {"INTERGENIC-3"}

STATE_GROUP_LABELS = {
    ("progenitor_side", "cells"): "Progenitor + Progenitor\nCycling cell count",
    ("differentiated", "cells"): "Differentiated cell count",
    (
        "progenitor_side",
        "clones",
    ): "# clones with at least one\nProgenitor or Progenitor Cycling cell",
    ("differentiated", "clones"): "# clones with at least one\ndifferentiated cell",
}

PANEL_FIG_SIZE = (10, 8.75)


@dataclass
class ObsOnlyData:
    obs: pd.DataFrame

    @property
    def n_obs(self) -> int:
        return self.obs.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../manuscript/plots/Sup_Figure_4_4",
    )
    parser.add_argument("--panel-letter", required=True)
    parser.add_argument(
        "--analysis-type",
        choices=["cells", "clones"],
        required=True,
    )
    parser.add_argument(
        "--state-group",
        choices=["progenitor_side", "differentiated"],
        required=True,
    )
    return parser.parse_args()


def subset_state_group(obs: pd.DataFrame, state_group: str) -> ObsOnlyData:
    if state_group == "progenitor_side":
        mask = obs["cell_type"].isin(PROGENITOR_SIDE)
    else:
        mask = ~obs["cell_type"].isin(PROGENITOR_SIDE)
    return ObsOnlyData(obs=obs.loc[mask].copy())


def main() -> None:
    args = parse_args()

    print("Loading gene10 data...")
    adata = get_adata(
        dataset="gene_10",
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=False,
    )
    adata.obs["guide"] = adata.obs["guide"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    obs = adata.obs.loc[~adata.obs["guide"].isin(GUIDES_TO_REMOVE)].copy()
    del adata
    guide_names = sorted(obs["guide"].unique())

    print(f"Subsetting to state group: {args.state_group}")
    subset = subset_state_group(obs, args.state_group)
    print(f"Subset contains {subset.n_obs} cells across {len(guide_names)} guides")

    print("Loading guide concentration data...")
    guide_conc_df = pd.read_csv(
        "../data/guide_concentration/gene10.csv"
    )
    guide_conc_df["short_guide"] = guide_conc_df["Row Barcode IDs"].apply(convert_long_to_short)
    guide_conc_dict = guide_conc_df.set_index("short_guide")["cp8 raw counts"].to_dict()

    perform_analysis(
        adata=subset,
        guide_conc_dict=guide_conc_dict,
        dataset_name=f"gene10_{args.state_group}",
        analysis_type=args.analysis_type,
        figure_num="4",
        panel_letter=args.panel_letter,
        table_suffix=None,
        remove_guides=None,
        x_scale_factor=1e4,
        x_axis_label="initial guide abundance (reads x 10^4)",
        xtick_values=[1, 2, 3, 4, 5],
        output_dir=args.output_dir,
        y_axis_label=STATE_GROUP_LABELS[(args.state_group, args.analysis_type)],
        guide_names=guide_names,
        fig_size=PANEL_FIG_SIZE,
    )

    print("\ndeprecated internal Sup_Figure_4_4 panel completed successfully!")


if __name__ == "__main__":
    main()
