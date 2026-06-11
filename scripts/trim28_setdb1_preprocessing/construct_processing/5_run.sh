# create umi_tools conda environment
# conda create -n umi_tools_env -c bioconda python=3.7 umi_tools
# source activate umi_tools_env
# Make sure to run conda activate pycashier

base_folder="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/"
fastq_folder=${base_folder}"/fastq_merged/"

# sampleArray=("lane1" "lane2" "lane3" "lane4" "lane5" "lane6" "lane7" "lane8" "lane9" "lane10" "lane11" "lane12" "lane13" "lane14" "lane15" "lane16")
sampleArray=("lane15" "lane16")

results_folder=${base_folder}"results/"
mkdir ${results_folder}
mkdir ${results_folder}/"umi_tools"

#INPUT REQUIRED
reference_folder="/Genomics/pritykinlab/share/perturbseq/fromMarc/new_data/trim28_setdb1_data/reference_genome"

umi_tools="/Genomics/pritykinlab/dillon/software/miniconda/envs/envs/umi_tools/bin/umi_tools"
STAR="/Genomics/pritykinlab/dillon/software/miniconda/envs/envs/STAR/bin/STAR"
samtools="/Genomics/pritykinlab/dillon/software/miniconda/envs/envs/samtools/bin/samtools"
pycashier="/Genomics/pritykinlab/dillon/software/miniconda/envs/envs/pycashier/bin/pycashier"

for sample in ${sampleArray[*]}; do
    results_folder=${base_folder}"results/"

     ##############################################################

     # gRNA MAPPING
     R1_file=${sample}"_R1.fastq.gz"
     R1_file_extracted=${sample}".unmapped_R1_extracted.fastq.gz"
     R2_pre=${sample}"_R2.fastq.gz"
     R2_file=${sample}".unmapped_R2.20bp_5prime.fq.gz"
     R2_file_extracted=${sample}".unmapped_R2_extracted.fastq.gz"

     # first make output folder
     output_folder=${results_folder}/"umi_tools"/${sample}
     mkdir ${output_folder}
     cd ${output_folder}

     /Genomics/pritykinlab/dillon/software/miniconda/envs/envs/scvi/bin/python3 \
          /Genomics/pritykinlab/dillon/perturbseq/scripts/trim28_setdb1_analysis/preprocessing_scripts/trim.py \
          ${fastq_folder}/${R2_pre} ${output_folder}/${R2_file} 55 75

     # generate whitelist for cell barcodes
     whitelist="whitelist.txt"

     ${umi_tools} whitelist --stdin ${fastq_folder}/${R1_file} \
                         --bc-pattern=CCCCCCCCCCCCCCCCNNNNNNNNNN \
                         --plot-prefix \
                         --method umis \
                         --set-cell-number 25000 \
                         --log2stderr > ${whitelist} \
                         -L whitelist.log \
                         --ed-above-threshold=discard

     # extracted R2 with cellbarcode tag
     ${umi_tools} extract --bc-pattern=CCCCCCCCCCCCCCCCNNNNNNNNNN \
                  --stdin ${fastq_folder}/${R1_file} \
                  --stdout ${fastq_folder}/${R1_file_extracted} \
                  --read2-in ${output_folder}/${R2_file} \
                  --read2-out=${R2_file_extracted} \
                  --error-correct-cell \
                  --whitelist=${whitelist} \
                  -L extract.log

                  # --filter-cell-barcode \

     # align reads to reference

     ${STAR} --runThreadN 20 \
          --genomeDir ${reference_folder}/STAR \
          --readFilesIn ${R2_file_extracted} \
          --outFilterMultimapNmax 1 \
          --outSAMtype BAM SortedByCoordinate \
          --readFilesCommand zcat \
          --limitBAMsortRAM 41143265264

     # --readMapNumber 100000 \

     # create feature count matrix

     count_matrix_wide=${sample}"_barcode_counts_wide_d3r2.tsv.gz"
     count_matrix_long=${sample}"_barcode_counts_long_d3r2.tsv.gz"

     /Genomics/pritykinlab/dillon/software/miniconda/envs/envs/subread/bin/featureCounts -O -T 8 -a ${reference_folder}/annotation_barcodes.gtf -o gene_assigned.txt -R BAM Aligned.sortedByCoord.out.bam

     ${samtools} sort Aligned.sortedByCoord.out.bam.featureCounts.bam -o assigned_sorted.bam
     ${samtools} index assigned_sorted.bam

     ${umi_tools} count --per-gene --gene-tag=XT --assigned-status-tag=XS --per-cell --wide-format-cell-counts -I assigned_sorted.bam -S ${output_folder}/${count_matrix_wide}

     ${umi_tools} count --per-gene --gene-tag=XT --assigned-status-tag=XS --per-cell -I assigned_sorted.bam -S ${output_folder}/${count_matrix_long}

     # gRNA MAPPING
     R1_file=${sample}"_R1.fastq.gz"
     R1_file_extracted=${sample}".unmapped_untrimmed_R1_extracted.fastq.gz"
     R2_file=${sample}"_R2.fastq.gz"
     R2_file_extracted=${sample}".unmapped_untrimmed_R2_extracted.fastq.gz"
     whitelist_folder=${results_folder}/"umi_tools"/${sample}
     whitelist=${whitelist_folder}/"whitelist.txt"

     output_folder=${results_folder}/"umi_tools"/${sample}
     cd ${output_folder}

     # extracted R2 with cellbarcode tag
     ${umi_tools} extract --bc-pattern=CCCCCCCCCCCCCCCCNNNNNNNNNN \
                  --stdin ${fastq_folder}/${R1_file} \
                  --stdout ${fastq_folder}/${R1_file_extracted} \
                  --read2-in ${fastq_folder}/${R2_file} \
                  --read2-out=${fastq_folder}/${R2_file_extracted} \
                  --error-correct-cell \
                  --whitelist=${whitelist} \
                  -L extract.log

     mv ${fastq_folder}/${R2_file_extracted} ${output_folder}

    results_folder=${base_folder}"results/pycashier_umi"
    mkdir ${results_folder}

     R2_file_extracted=${sample}".unmapped_untrimmed_R2_extracted.fastq.gz"

     # use pycahsier to extract UMIs

     # run following command to extract 10bp UMI
     # first make output folder
     fastq_folder=${base_folder}"results/umi_tools"/${sample}
     mkdir -p ${results_folder}/${sample}/input
     mkdir ${results_folder}/${sample}/output
     cp ${fastq_folder}/${R2_file_extracted} ${results_folder}/${sample}/input

     cd ${results_folder}/${sample}/input

     gzip -d *fastq.gz

     cd ${results_folder}/${sample}/output

     # run following command to extract 10bp UMI
     ${pycashier} extract -y -i ${results_folder}/${sample}/input -o ${results_folder}/${sample}/output -t 10 -ua TGAACTTGCTA -da TAGCTCTAAAC -d 1 -r 3 -l 6 -fc 3
     # pycashier ${output_folder} -sc -t 10 -ua TTGAGAGCTA -da TAGCAAGTTC -d 1 -r 3 -bl 10 -fc 3
done
