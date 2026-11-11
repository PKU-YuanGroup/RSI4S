# Genomic Sequence Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K9me3, H4K20me1, deepstarr, enhancers, enhancers_types, promoter_all, promoter_no_tata, promoter_tata, splice_sites_acceptors, splice_sites_all, splice_sites_donors. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Understanding how DNA sequences encode molecular phenotypes is a fundamental challenge in genomics. DNA sequences contain complex patterns that determine gene regulation, chromatin organization, and RNA processing. Predicting these functional properties from raw nucleotide sequences has broad applications in understanding gene regulation, interpreting genetic variants, and characterizing regulatory elements across different organisms.

This task encompasses 19 diverse genomic prediction problems spanning chromatin profile prediction (identifying histone modifications from DNA sequence), regulatory element identification (detecting enhancers and their subtypes), promoter recognition (distinguishing TATA-box and non-TATA promoters), splice site detection (identifying splice acceptors and donors), and enhancer activity quantification (predicting the strength of developmental and housekeeping enhancer activity from *Drosophila melanogaster* sequences). The breadth of these tasks tests the ability of computational methods to capture diverse sequence-function relationships across different genomic contexts.

## 2. Task Objective

Given a DNA sequence, predict its associated genomic property.

*   **Input**: Raw DNA nucleotide sequences (strings of A, T, C, G characters). Sequence lengths vary by task category: 1,000 bp for chromatin profiles, 400 bp for enhancer elements, 300 bp for promoters, 600 bp for splice sites, and 249 bp for enhancer activity regression.
*   **Output**: For classification tasks (18 instances): integer class labels (binary 0/1 for most tasks; ternary 0/1/2 for `enhancers_types` and `splice_sites_all`). For the regression task (`deepstarr`): two continuous values per sequence representing developmental and housekeeping enhancer activity.

## 4. Evaluation Metrics

### Classification Instances (18 tasks)

#### Primary Metric
*   **Name**: MCC (Matthews Correlation Coefficient)
*   **Definition**: A balanced measure of classification quality that accounts for true/false positives and negatives. Ranges from -1 to +1, where +1 indicates perfect prediction, 0 indicates random prediction, and -1 indicates total disagreement. For multiclass tasks (enhancers_types, splice_sites_all), the multiclass generalization of MCC is used. Higher is better.

### Regression Instance (deepstarr)

#### Primary Metric
*   **Name**: Pearson_Dev (Pearson Correlation - Developmental)
*   **Definition**: Pearson correlation coefficient between predicted and true developmental enhancer activity (Dev_log2_enrichment). Ranges from -1 to +1. Higher is better.

#### Other Metrics
*   **Pearson_Hk** (Pearson Correlation - Housekeeping): Pearson correlation coefficient between predicted and true housekeeping enhancer activity (Hk_log2_enrichment). Higher is better.
