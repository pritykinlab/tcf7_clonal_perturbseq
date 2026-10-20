# Data Inputs

## GEO downloads

The study's sequencing datasets are deposited in GEO:

| Dataset | GEO record | Processed-file directory |
| --- | --- | --- |
| Bulk RNA-seq | [GSE334845](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334845) | [Normalized count tables](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE334nnn/GSE334845/suppl/) |
| Pooled bulk CRISPR screens | [GSE334846](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334846) | [Screen tables](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE334nnn/GSE334846/suppl/) |
| 10-gene Perturb-seq | [GSE334877](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334877) | See the GEO record; public download access was not verified on October 20, 2026. |
| Trim28/Setdb1 Perturb-seq | [GSE334890](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334890) | [AnnData and feature reference](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE334nnn/GSE334890/suppl/) |
| Trim28 single-cell multiome | [GSE334878](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334878) | [RNA and ATAC AnnData files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE334nnn/GSE334878/suppl/) |

Download the processed AnnData files and place them at the workflow input paths:

| GEO filename | Local path |
| --- | --- |
| `GSE334890_trim28_setdb1_final.h5ad` | `data/trim28_setdb1/preprocessing/final.h5ad` |
| `GSE334878_multiome_rna_final.h5ad` | `data/multiome/analysis_using_atlas_v2/rna_final.h5ad` |
| `GSE334878_multiome_atac_final.h5ad` | `data/multiome/analysis_using_atlas_v2/atac_final.h5ad` |

The 10-gene processed AnnData input is expected at
`data/gene10/preprocessing/final.h5ad`. Obtain it through GSE334877 when
available, or obtain reviewer access from the corresponding authors if the
record remains private. Raw sequencing reads are linked from GEO through SRA.
These downloads provide study data; the full workflows also require the
additional inputs listed below.

## Workflow inputs

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
