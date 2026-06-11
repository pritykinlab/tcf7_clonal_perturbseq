GENE10_ADATA = "data/gene10/preprocessing/final.h5ad"
TRIM28_SETDB1_ADATA = "data/trim28_setdb1/preprocessing/final.h5ad"

GENE10_PERTURBATIONS = [
    "NT",
    "Bach2",
    "Ifngr1",
    "Ifngr2",
    "Il18r1",
    "Jak2",
    "Jak3",
    "Prdm1",
    "Setdb1",
    "Tnfrsf1b",
    "Trim28",
]
GENE10_PLOT_PERTURBATIONS = ["NT", "Bach2"]
TRIM28_PERTURBATIONS = ["NT", "Trim28", "Setdb1"]
TRIM28_PLOT_PERTURBATIONS = ["NT", "Trim28"]


rule export_clone_counts_gene10:
    """Build clone count inputs from the processed 10-gene AnnData file."""
    input:
        adata=GENE10_ADATA
    output:
        expand(f"{REAL_DATA_DIR}/gene10/input/{{perturbation}}.csv", perturbation=GENE10_PERTURBATIONS)
    params:
        perturbations=",".join(GENE10_PERTURBATIONS),
        output_dir=f"{REAL_DATA_DIR}/gene10/input"
    shell:
        """
        mkdir -p {params.output_dir}
        {PYTHON} clonal_analysis/scripts/export_clone_counts_from_adata.py \
            --dataset gene10 \
            --adata {input.adata} \
            --output-dir {params.output_dir} \
            --perturbations {params.perturbations}
        """


rule export_clone_counts_trim28_setdb1:
    """Build clone count inputs from the processed Trim28/Setdb1 AnnData file."""
    input:
        adata=TRIM28_SETDB1_ADATA
    output:
        expand(f"{REAL_DATA_DIR}/trim28_setdb1/input/{{perturbation}}.csv", perturbation=TRIM28_PERTURBATIONS)
    params:
        perturbations=",".join(TRIM28_PERTURBATIONS),
        output_dir=f"{REAL_DATA_DIR}/trim28_setdb1/input"
    shell:
        """
        mkdir -p {params.output_dir}
        {PYTHON} clonal_analysis/scripts/export_clone_counts_from_adata.py \
            --dataset trim28_setdb1 \
            --adata {input.adata} \
            --output-dir {params.output_dir} \
            --perturbations {params.perturbations}
        """


rule compute_loss_landscape_gene10:
    """Compute a parameter-loss grid for one 10-gene perturbation."""
    input:
        real_data=f"{REAL_DATA_DIR}/gene10/input/{{perturbation}}.csv"
    output:
        f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/grid_search_loss.npz"
    params:
        dataset_key=lambda wildcards: f"gene10_{wildcards.perturbation}"
    wildcard_constraints:
        perturbation="|".join(GENE10_PERTURBATIONS)
    shell:
        """
        mkdir -p {CLONAL_OUTPUT_DIR}/gene10/{wildcards.perturbation}
        {PYTHON} clonal_analysis/scripts/compute_loss_landscape.py \
            --dataset {params.dataset_key} \
            --output {output} \
            --mean-divisions 5 \
            --subsample-fraction 0.15
        """


rule compute_loss_landscape_trim28_setdb1:
    """Compute a parameter-loss grid for one Trim28/Setdb1 perturbation."""
    input:
        real_data=f"{REAL_DATA_DIR}/trim28_setdb1/input/{{perturbation}}.csv"
    output:
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/grid_search_loss.npz"
    params:
        dataset_key=lambda wildcards: f"trim28_setdb1_{wildcards.perturbation}"
    wildcard_constraints:
        perturbation="|".join(TRIM28_PERTURBATIONS)
    shell:
        """
        mkdir -p {CLONAL_OUTPUT_DIR}/trim28_setdb1/{wildcards.perturbation}
        {PYTHON} clonal_analysis/scripts/compute_loss_landscape.py \
            --dataset {params.dataset_key} \
            --output {output} \
            --mean-divisions 5 \
            --subsample-fraction 0.1
        """


rule plot_lowest_loss_points_3d_gene10:
    """Generate an interactive 3D plot of the lowest-loss NT and Bach2 parameter sets."""
    input:
        nt=f"{CLONAL_OUTPUT_DIR}/gene10/NT/grid_search_loss.npz",
        bach2=f"{CLONAL_OUTPUT_DIR}/gene10/Bach2/grid_search_loss.npz"
    output:
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_3d.html"
    threads: 4
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_points_3d.py \
            --nt-data {input.nt} \
            --bach2-data {input.bach2} \
            --output {output} \
            --n-points 1000 \
            --n-cpus {threads}
        """


rule plot_lowest_loss_points_3d_trim28_setdb1:
    """Generate an interactive 3D plot for NT and Trim28 in the Trim28/Setdb1 dataset."""
    input:
        nt=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/NT/grid_search_loss.npz",
        trim28=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/Trim28/grid_search_loss.npz"
    output:
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_3d.html"
    threads: 4
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_points_3d_multi.py \
            --output {output} \
            --n-points 1000 \
            --n-cpus {threads} \
            --subset-min 5 \
            --subset-step 5 \
            --dataset NT_trim28_setdb1={input.nt} \
            --dataset Trim28={input.trim28}
        """


rule plot_lowest_loss_points_multi_view_gene10:
    """Generate static multi-view projections for NT and Bach2 parameter sets."""
    input:
        nt=f"{CLONAL_OUTPUT_DIR}/gene10/NT/grid_search_loss.npz",
        bach2=f"{CLONAL_OUTPUT_DIR}/gene10/Bach2/grid_search_loss.npz"
    output:
        png=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view.png",
        projections=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections.png",
        txt=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_equations.txt"
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_points_multi_view.py \
            --dataset NT={input.nt} \
            --dataset Bach2={input.bach2} \
            --output-prefix {CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view \
            --n-points 100 \
            --highlight NT
        """


rule plot_lowest_loss_points_multi_view_gene10_legend:
    """Generate the standalone legend for the gene10 projection views."""
    input:
        nt=f"{CLONAL_OUTPUT_DIR}/gene10/NT/grid_search_loss.npz",
        bach2=f"{CLONAL_OUTPUT_DIR}/gene10/Bach2/grid_search_loss.npz"
    output:
        png=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections_legend.png",
        svg=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections_legend.svg"
    params:
        output_prefix=f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections_legend"
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_points_legend.py \
            --dataset NT={input.nt} \
            --dataset Bach2={input.bach2} \
            --output-prefix {params.output_prefix} \
            --n-points 100 \
            --highlight NT
        """


rule plot_lowest_loss_points_multi_view_trim28_setdb1:
    """Generate static multi-view projections for NT and Trim28 parameter sets."""
    input:
        nt=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/NT/grid_search_loss.npz",
        trim28=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/Trim28/grid_search_loss.npz"
    output:
        png=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view.png",
        projections=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view_projections.png",
        txt=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view_equations.txt"
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_points_multi_view.py \
            --dataset NT={input.nt} \
            --dataset Trim28={input.trim28} \
            --output-prefix {CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view \
            --n-points 100 \
            --skip-highlight
        """


rule plot_lowest_loss_gene10_diagnostics:
    """Generate ECDF and isotonic diagnostics for one 10-gene perturbation."""
    input:
        grid=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/grid_search_loss.npz",
        real_data=f"{REAL_DATA_DIR}/gene10/input/{{perturbation}}.csv"
    output:
        ecdf_png=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.png",
        ecdf_svg=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.svg",
        iso_png=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.png",
        iso_svg=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.svg",
        params=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_parameters.txt"
    params:
        output_prefix=f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}"
    wildcard_constraints:
        perturbation="|".join(GENE10_PERTURBATIONS)
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_nt_diagnostics.py \
            --grid {input.grid} \
            --label {wildcards.perturbation} \
            --real-data {input.real_data} \
            --output-prefix {params.output_prefix} \
            --mean-divisions 5 \
            --subsample-fraction 0.15
        """


rule plot_lowest_loss_trim28_setdb1_diagnostics:
    """Generate ECDF and isotonic diagnostics for one Trim28/Setdb1 perturbation."""
    input:
        grid=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/grid_search_loss.npz",
        real_data=f"{REAL_DATA_DIR}/trim28_setdb1/input/{{perturbation}}.csv"
    output:
        ecdf_png=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.png",
        ecdf_svg=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.svg",
        iso_png=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.png",
        iso_svg=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.svg",
        params=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_parameters.txt"
    params:
        output_prefix=f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}"
    wildcard_constraints:
        perturbation="|".join(TRIM28_PERTURBATIONS)
    shell:
        """
        {PYTHON} clonal_analysis/scripts/plot_lowest_loss_nt_diagnostics.py \
            --grid {input.grid} \
            --label {wildcards.perturbation} \
            --real-data {input.real_data} \
            --output-prefix {params.output_prefix} \
            --mean-divisions 5 \
            --subsample-fraction 0.1
        """


rule bar_plot_parameters_gene10:
    """Compare permissible parameter ranges across the 10-gene perturbations."""
    input:
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/grid_search_loss.npz", perturbation=GENE10_PERTURBATIONS)
    output:
        f"{CLONAL_OUTPUT_DIR}/gene10/parameter_ranges_bar_plot.png"
    params:
        dataset_args=" ".join(
            f"--dataset {perturbation}={CLONAL_OUTPUT_DIR}/gene10/{perturbation}/grid_search_loss.npz"
            for perturbation in GENE10_PERTURBATIONS
        )
    shell:
        """
        {PYTHON} clonal_analysis/scripts/bar_plot_parameters.py \
            {params.dataset_args} \
            --output {output} \
            --n-points 50
        """


rule bar_plot_parameters_trim28_setdb1:
    """Compare permissible parameter ranges across Trim28/Setdb1 perturbations."""
    input:
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/grid_search_loss.npz", perturbation=TRIM28_PERTURBATIONS)
    output:
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/parameter_ranges_bar_plot.png"
    params:
        dataset_args=" ".join(
            f"--dataset {perturbation}={CLONAL_OUTPUT_DIR}/trim28_setdb1/{perturbation}/grid_search_loss.npz"
            for perturbation in TRIM28_PERTURBATIONS
        )
    shell:
        """
        {PYTHON} clonal_analysis/scripts/bar_plot_parameters.py \
            {params.dataset_args} \
            --output {output} \
            --n-points 50
        """


rule gene10_all_perturbation_diagnostics:
    """Generate all 10-gene clonal-model fits and diagnostics."""
    input:
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.png", perturbation=GENE10_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.svg", perturbation=GENE10_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.png", perturbation=GENE10_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.svg", perturbation=GENE10_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/gene10/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_parameters.txt", perturbation=GENE10_PERTURBATIONS),
        f"{CLONAL_OUTPUT_DIR}/gene10/parameter_ranges_bar_plot.png"


rule trim28_setdb1_all_perturbation_diagnostics:
    """Generate all Trim28/Setdb1 clonal-model fits and diagnostics."""
    input:
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.png", perturbation=TRIM28_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_ecdf.svg", perturbation=TRIM28_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.png", perturbation=TRIM28_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_isotonic.svg", perturbation=TRIM28_PERTURBATIONS),
        expand(f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/{{perturbation}}/lowest_loss_points_best_{{perturbation}}_parameters.txt", perturbation=TRIM28_PERTURBATIONS),
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/parameter_ranges_bar_plot.png"


rule all_clonal_modeling:
    """Generate the modeled clonal landscapes and diagnostics used by the figures."""
    input:
        rules.gene10_all_perturbation_diagnostics.input,
        rules.trim28_setdb1_all_perturbation_diagnostics.input,
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_3d.html",
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view.png",
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections.png",
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_equations.txt",
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections_legend.png",
        f"{CLONAL_OUTPUT_DIR}/gene10/lowest_loss_points_multi_view_projections_legend.svg",
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_3d.html",
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view.png",
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view_projections.png",
        f"{CLONAL_OUTPUT_DIR}/trim28_setdb1/lowest_loss_points_multi_view_equations.txt"
