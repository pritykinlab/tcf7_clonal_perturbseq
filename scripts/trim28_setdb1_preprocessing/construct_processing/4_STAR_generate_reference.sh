#!/bin/bash

STAR="/Genomics/pritykinlab/dillon/software/miniconda/envs/envs/STAR/bin/STAR"
combined_genome="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/reference_genome/combined_genome.fa"
genomeDIR="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/reference_genome/STAR"
gtf="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/reference_genome/annotation_barcodes.gtf"
number_of_threads=20

$STAR --runMode genomeGenerate \
--runThreadN $number_of_threads \
--genomeFastaFiles $combined_genome \
--genomeDir $genomeDIR \
--sjdbGTFfile $gtf \
--limitGenomeGenerateRAM 35952560000 \
--genomeSAindexNbases 3 \
--sjdbOverhang 19
