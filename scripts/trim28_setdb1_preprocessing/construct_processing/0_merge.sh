#!/bin/bash

# Defining variables and paths
base_folder="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/fastq_all"

results_folder="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/fastq_merged"

cd ${base_folder}

# Lane array containing the lane information
laneArray=("lane1" "lane2" "lane3" "lane4" "lane5" "lane6" "lane7" "lane8" "lane9" "lane10" "lane11" "lane12" "lane13" "lane14" "lane15" "lane16")

for lane in ${laneArray[*]}; do
    cat *${lane}*_R1_001.fastq.gz > ${results_folder}/${lane}_R1.fastq.gz &
    cat *${lane}*_R2_001.fastq.gz > ${results_folder}/${lane}_R2.fastq.gz
done
