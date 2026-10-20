# tcf7_clonal_perturbseq

Reproducibility archive for the Tcf7 clonal Perturb-seq manuscript analyses.
The repository contains source scripts and workflow files for all manuscript
figures, public supplementary figures, supplementary tables, clonal modeling,
and preprocessing notebooks for the 10-gene and Trim28/Setdb1 Perturb-seq
datasets.

## Data availability

Raw and processed sequencing data generated in this study have been deposited
in the NCBI Gene Expression Omnibus (GEO):

| Dataset | GEO accession |
| --- | --- |
| Bulk RNA-seq | [GSE334845](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334845) |
| Pooled bulk CRISPR screens | [GSE334846](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334846) |
| 10-gene Perturb-seq | [GSE334877](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334877) |
| Trim28/Setdb1 Perturb-seq | [GSE334890](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334890) |
| Trim28 single-cell multiome | [GSE334878](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE334878) |

Processed files are available through the supplementary-file links on released
GEO records; raw sequencing reads are linked through SRA. See
[data/README.md](data/README.md) for download locations and the local paths
expected by the workflows. Records that have not yet been released require
reviewer access from the corresponding authors.

## System requirements

The analyses were developed and run on the Linux cluster. The current
host runs Springdale Open Enterprise Linux 8.8 (Modena), x86_64.
For full-dataset analyses, plan for approximately 256 GB RAM; this is a
recommended resource budget. Python 3.10 and the
analysis dependencies are specified in [environment.yml](environment.yml).

Estimated runtime for all analyses is approximately 24 hours on a system
meeting the requirements above. Actual runtime depends on hardware and
parallelization.

## Setup

Create the analysis environment:

```bash
conda env create -f environment.yml
conda activate main_scanpy
```

Place large processed inputs in the paths documented in `data/README.md`.

## Notebooks

Preprocessing notebooks are organized under `scripts/gene10_preprocessing/`
and `scripts/trim28_setdb1_preprocessing/`.

## License

The software in this repository is released under the [MIT License](LICENSE).
