import os

PYTHON = os.environ.get("PYTHON", "python")
SNAKEMAKE = os.environ.get("SNAKEMAKE", "snakemake")
PLOTS_DIR = "manuscript/plots"
SCRIPTS_DIR = "manuscript/scripts"
SUPPLEMENTARY_TABLE_DIR = "manuscript/supplementary_tables"
CLONAL_OUTPUT_DIR = "clonal_analysis/intermediate_files"
REAL_DATA_DIR = f"{CLONAL_OUTPUT_DIR}/real_data"

PYTHONPATH = ":".join([
    "scripts/utils",
    "manuscript/scripts",
    "clonal_analysis/scripts",
])

shell.prefix(f"export PYTHONPATH='{PYTHONPATH}:$PYTHONPATH'; ")


rule all:
    input:
        "workflow/.all_reproducibility_archive.done"


include: "workflow/rules/clonal_modeling.smk"
include: "workflow/rules/notebooks.smk"


rule all_manuscript:
    output:
        touch("workflow/.all_manuscript.done")
    threads: 1
    shell:
        """
        cd manuscript && {SNAKEMAKE} all --cores {threads} -p
        """


rule all_reproducibility_archive:
    input:
        rules.all_clonal_modeling.input,
        rules.execute_notebooks.output,
        "workflow/.all_manuscript.done"
    output:
        touch("workflow/.all_reproducibility_archive.done")
