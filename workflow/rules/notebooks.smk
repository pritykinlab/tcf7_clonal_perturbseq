import glob

NOTEBOOK_INPUTS = sorted(
    glob.glob("scripts/gene10_preprocessing/*.ipynb")
    + glob.glob("scripts/trim28_setdb1_preprocessing/*.ipynb")
    + glob.glob("scripts/trim28_setdb1_preprocessing/construct_processing/*.ipynb")
)


rule execute_notebooks:
    """Execute archival notebooks into ignored result copies without changing committed notebooks."""
    input:
        NOTEBOOK_INPUTS
    output:
        touch("results/notebooks/.executed")
    params:
        notebooks=" ".join(NOTEBOOK_INPUTS)
    shell:
        """
        for nb in {params.notebooks}; do
            out_dir="results/notebooks/$(dirname "$nb")"
            mkdir -p "$out_dir"
            jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$nb")" --output-dir "$out_dir"
        done
        """
