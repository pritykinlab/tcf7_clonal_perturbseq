# tcf7_clonal_perturbseq

Reproducibility archive for the Tcf7 clonal Perturb-seq manuscript analyses.
The repository contains source scripts and workflow files for all manuscript
figures, public supplementary figures, supplementary tables, clonal modeling,
and preprocessing notebooks for the 10-gene and Trim28/Setdb1 Perturb-seq
datasets.

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
