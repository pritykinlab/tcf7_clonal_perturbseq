#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c("DESeq2")
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      "Missing required R packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install DESeq2 before running this rule.",
      call. = FALSE
    )
  }
  library(DESeq2)
})

parse_args <- function(argv) {
  args <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--") || i == length(argv)) {
      stop("Arguments must be provided as --name value pairs.", call. = FALSE)
    }
    args[[substring(key, 3)]] <- argv[[i + 1]]
    i <- i + 2
  }
  args
}

require_arg <- function(args, key) {
  value <- args[[key]]
  if (is.null(value) || !nzchar(value)) {
    stop("Missing required argument --", key, call. = FALSE)
  }
  value
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
sample_table_path <- require_arg(args, "sample-table")
tx2gene_path <- require_arg(args, "tx2gene")
gene_metadata_path <- require_arg(args, "gene-metadata")
normalized_counts_path <- require_arg(args, "normalized-counts-output")
vst_expression_path <- require_arg(args, "vst-expression-output")
pca_output_path <- args[["pca-output"]]
if (is.null(pca_output_path)) {
  pca_output_path <- args[["figure1c-pca-output"]]
}
write_pca_output <- !is.null(pca_output_path) && nzchar(pca_output_path)
size_factors_path <- require_arg(args, "size-factors-output")
session_info_path <- require_arg(args, "session-info-output")

sample_table <- read.delim(sample_table_path, check.names = FALSE, stringsAsFactors = FALSE)
tx2gene <- read.delim(tx2gene_path, check.names = FALSE, stringsAsFactors = FALSE)
gene_metadata <- read.delim(gene_metadata_path, check.names = FALSE, stringsAsFactors = FALSE)

main_groups <- c("Naive", "Lenti", "Activated", "D7 Control", "D7 Lenti", "D7 Activated")
memory_groups <- c(
  "D30 Control CM",
  "D30 Control EM",
  "D30 Lenti CM",
  "D30 Lenti EM",
  "D30 Activated CM",
  "D30 Activated EM"
)
pca_groups <- c(main_groups, memory_groups)

clean_sampletype <- function(sampletype) {
  if (sampletype == "Naive") {
    return("Naive")
  }
  if (sampletype %in% c("Thy1.1pos", "Thy1.1neg")) {
    return("Lenti")
  }
  if (sampletype == "Tcellsstim") {
    return("Activated")
  }
  if (sampletype %in% c("D8_naive_control", "D8_activated_control")) {
    return("D7 Control")
  }
  if (sampletype %in% c("D8_naive_Thy1.1pos", "D8_naive_Thy1.1neg")) {
    return("D7 Lenti")
  }
  if (sampletype %in% c("D8_activated_Thy1.1pos", "D8_activated_Thy1.1neg")) {
    return("D7 Activated")
  }
  if (sampletype %in% c(
    "D30_naive_control_CM",
    "D30_fresh_Thy1.1_CM",
    "D30_fresh_control_CM",
    "D30_activated_control_CM"
  )) {
    return("D30 Control CM")
  }
  if (sampletype %in% c(
    "D30_naive_control_EM",
    "D30_fresh_Thy1.1_EM",
    "D30_fresh_control_EM",
    "D30_activated_control_EM"
  )) {
    return("D30 Control EM")
  }
  if (sampletype == "D30_naive_Thy1.1_CM") {
    return("D30 Lenti CM")
  }
  if (sampletype == "D30_naive_Thy1.1_EM") {
    return("D30 Lenti EM")
  }
  if (sampletype == "D30_activated_Thy1.1_CM") {
    return("D30 Activated CM")
  }
  if (sampletype == "D30_activated_Thy1.1_EM") {
    return("D30 Activated EM")
  }
  NA_character_
}

required_sample_columns <- c("sample", "sampletype", "quant_file")
missing_sample_columns <- setdiff(required_sample_columns, colnames(sample_table))
if (length(missing_sample_columns) > 0) {
  stop("Sample table is missing columns: ", paste(missing_sample_columns, collapse = ", "), call. = FALSE)
}

missing_quant_files <- sample_table$quant_file[!file.exists(sample_table$quant_file)]
if (length(missing_quant_files) > 0) {
  stop(
    "Missing ", length(missing_quant_files), " Salmon quant.sf files. First missing path: ",
    missing_quant_files[[1]],
    call. = FALSE
  )
}

files <- sample_table$quant_file
names(files) <- sample_table$sample

gene_ids <- gene_metadata$gene_id
count_matrix <- matrix(0, nrow = length(gene_ids), ncol = length(files))
rownames(count_matrix) <- gene_ids
colnames(count_matrix) <- names(files)

for (sample in names(files)) {
  quant <- read.delim(files[[sample]], check.names = FALSE, stringsAsFactors = FALSE)
  quant <- quant[, c("Name", "NumReads")]
  merged <- merge(quant, tx2gene[, c("transcript_id", "gene_id")], by.x = "Name", by.y = "transcript_id")
  gene_counts <- rowsum(merged$NumReads, merged$gene_id, reorder = FALSE)
  matched <- match(rownames(gene_counts), rownames(count_matrix))
  count_matrix[matched, sample] <- gene_counts[, 1]
}

count_matrix <- round(count_matrix)
storage.mode(count_matrix) <- "integer"

coldata <- data.frame(
  sampletype = sample_table$sampletype,
  sample_group = vapply(sample_table$sampletype, clean_sampletype, character(1)),
  row.names = sample_table$sample,
  check.names = FALSE
)

dds <- DESeqDataSetFromMatrix(countData = count_matrix, colData = coldata, design = ~ 1)
dds <- dds[rowSums(counts(dds)) > 0, ]
dds <- estimateSizeFactors(dds)

normalized_counts <- counts(dds, normalized = TRUE)
vst_dds <- vst(dds, blind = TRUE)
vst_expression <- assay(vst_dds)
gene_match <- match(rownames(normalized_counts), gene_metadata$gene_id)
gene_name <- gene_metadata$gene_name[gene_match]
gene_name[is.na(gene_name) | gene_name == ""] <- rownames(normalized_counts)[is.na(gene_name) | gene_name == ""]

normalized_counts_df <- data.frame(
  gene_name = gene_name,
  gene_id = rownames(normalized_counts),
  check.names = FALSE
)
normalized_counts_df <- cbind(normalized_counts_df, as.data.frame(normalized_counts, check.names = FALSE))

dir.create(dirname(normalized_counts_path), recursive = TRUE, showWarnings = FALSE)
write.table(
  normalized_counts_df,
  file = normalized_counts_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

vst_expression_df <- data.frame(
  gene_name = gene_name,
  gene_id = rownames(vst_expression),
  check.names = FALSE
)
vst_expression_df <- cbind(vst_expression_df, as.data.frame(vst_expression, check.names = FALSE))
write.table(
  vst_expression_df,
  file = vst_expression_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

pca_ntop <- min(500L, nrow(vst_dds))
if (write_pca_output) {
  pca_sample_table <- sample_table
  pca_sample_table$sample_group <- vapply(pca_sample_table$sampletype, clean_sampletype, character(1))
  unmapped_sampletypes <- sort(unique(pca_sample_table$sampletype[is.na(pca_sample_table$sample_group)]))
  if (length(unmapped_sampletypes) > 0) {
    stop(
      "Unmapped sampletype values for Figure S1a PCA: ",
      paste(unmapped_sampletypes, collapse = ", "),
      call. = FALSE
    )
  }
  pca_sample_table <- pca_sample_table[pca_sample_table$sample_group %in% pca_groups, ]
  pca_sample_table$sample_group <- factor(pca_sample_table$sample_group, levels = pca_groups)
  pca_sample_table$sample_order <- seq_len(nrow(pca_sample_table))
  pca_sample_table <- pca_sample_table[
    order(pca_sample_table$sample_group, pca_sample_table$sample_order),
    c("sample", "sampletype", "sample_group")
  ]
  pca_sample_table$sample_group <- as.character(pca_sample_table$sample_group)
  pca_sample_table$display_order <- seq_len(nrow(pca_sample_table))

  vst_pca <- vst_dds[, pca_sample_table$sample]
  pca_data <- plotPCA(vst_pca, intgroup = "sample_group", ntop = pca_ntop, returnData = TRUE)
  pca_match <- match(pca_sample_table$sample, pca_data$name)
  if (any(is.na(pca_match))) {
    stop("DESeq2 plotPCA did not return coordinates for all Figure S1a samples.", call. = FALSE)
  }
  pca_percent_var <- attr(pca_data, "percentVar") * 100
  pca_output <- cbind(
    pca_sample_table,
    pca_data[pca_match, c("PC1", "PC2")]
  )
  pca_output$PC1_variance_percent <- pca_percent_var[[1]]
  pca_output$PC2_variance_percent <- pca_percent_var[[2]]
  pca_output$pca_method <- "DESeq2::plotPCA"
  pca_output$pca_transform <- "vst(blind=TRUE)"
  pca_output$pca_ntop <- pca_ntop
  write.table(pca_output, file = pca_output_path, sep = "\t", quote = FALSE, row.names = FALSE)
}

size_factors <- sizeFactors(dds)
if (is.null(size_factors)) {
  size_factors <- rep(NA_real_, ncol(dds))
}
size_factor_df <- data.frame(
  sample = colnames(dds),
  sampletype = coldata[colnames(dds), "sampletype"],
  size_factor = as.numeric(size_factors),
  check.names = FALSE
)
write.table(size_factor_df, file = size_factors_path, sep = "\t", quote = FALSE, row.names = FALSE)

sink(session_info_path)
cat("Input counts: Salmon transcript NumReads summed to GENCODE gene IDs and rounded.\n")
cat("Normalization: DESeq2 median-ratio size factors from design ~ 1.\n")
cat("PCA transform: DESeq2 variance-stabilizing transformation with blind=TRUE.\n")
if (write_pca_output) {
  cat("PCA method: DESeq2::plotPCA on Figure S1a samples with ntop =", pca_ntop, ".\n")
} else {
  cat("PCA output: not written by this rule; Figure S1a coordinates are generated from the VST matrix.\n")
}
cat("Genes retained after nonzero-count filter:", nrow(dds), "\n")
cat("Samples:", ncol(dds), "\n\n")
print(sessionInfo())
sink()

message("Wrote DESeq2-normalized counts: ", normalized_counts_path)
message("Wrote DESeq2 variance-stabilized expression: ", vst_expression_path)
if (write_pca_output) {
  message("Wrote DESeq2 plotPCA coordinates: ", pca_output_path)
}
message("Wrote size factors: ", size_factors_path)
message("Wrote session info: ", session_info_path)
