#!/bin/bash

# conda activate fastqc

# Define the directories
DATA_DIR="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/fastq_test"
FASTQC_OUTPUT_DIR="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/fastqc"
MULTIQC_OUTPUT_DIR="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/multiqc"

# Define the FASTQ files
FILE_R1="GRNA_lane1_S1_L001_R1_001.fastq.gz"
FILE_R2="GRNA_lane1_S1_L001_R2_001.fastq.gz"

# Create output directories if they don't exist
mkdir -p $FASTQC_OUTPUT_DIR
mkdir -p $MULTIQC_OUTPUT_DIR

# Run FastQC on R1 and R2 files using 16 threads and specify the output directory
fastqc -t 16 -o $FASTQC_OUTPUT_DIR $DATA_DIR/$FILE_R1 $DATA_DIR/$FILE_R2

# Run MultiQC to aggregate FastQC reports and specify the output directory
multiqc -o $MULTIQC_OUTPUT_DIR $FASTQC_OUTPUT_DIR
