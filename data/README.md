# Data Inputs

This repository tracks small static input tables and workflow code only. Large
processed data files are required to run the full analysis but are intentionally
not committed.

Place the following files at these paths before running the Snakemake workflows:

- `data/gene10/preprocessing/final.h5ad`
- `data/trim28_setdb1/preprocessing/final.h5ad`
- `data/comp_screen1/final.h5ad`
- `data/multiome/analysis_using_atlas_v2/rna_final.h5ad`
- `data/multiome/analysis_using_atlas_v2/atac_final.h5ad`
- additional small support files under `data/misc/`, `data/signatures/`,
  `data/multiome/`, and `data/repeat_enrichment/` referenced by the manuscript
  Snakefile and scripts
- external shared inputs under `raw_data/`, preserving the original project
  prefix structure after `raw_data/`
- Google Cloud downloads under `gcloud/`, preserving bucket prefixes where
  practical
- Figure 1 handoff data under `Figure1_data_handoff/`
- Figure 7 handoff data under `Figure7_data_context_2026-05-19/`

The guide-concentration tables under `data/guide_concentration/` are tracked
because they are small static inputs used by the clonal abundance analyses.
