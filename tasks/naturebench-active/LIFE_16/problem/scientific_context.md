# Noncoding Variant Effect Prediction on Gene Expression

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: gtex_v8. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Noncoding genetic variants constitute the vast majority of human genetic variation and play critical roles in regulating gene expression through interactions with cis-regulatory elements such as enhancers, silencers, and promoters. Understanding how these variants affect gene expression across diverse tissues is fundamental to human genetics and precision medicine, yet remains one of the most pressing challenges in genomics.

A key difficulty lies in the context-dependent nature of regulatory effects: the same noncoding variant may have different impacts across tissues due to differences in chromatin accessibility and the regulatory landscape. Accurately predicting both the direction (up- or down-regulation) and magnitude of expression changes induced by noncoding single nucleotide polymorphisms (SNPs) would enable deeper understanding of complex disease mechanisms and inform therapeutic strategies.

## 2. Task Objective

Given a noncoding SNP-gene pair along with the associated DNA sequence context and tissue-specific chromatin accessibility (ATAC-seq) data, predict:

*   **Classification**: The direction of the SNP's regulatory effect on gene expression -- whether it causes up-regulation (label = 1) or down-regulation (label = 0).
*   **Regression**: The magnitude of the regulatory effect, measured as the eQTL slope value (a continuous value where positive indicates up-regulation and negative indicates down-regulation).

**Input**: For each test sample, the following features are provided:
- Reference and mutant 51-bp DNA sequences centered on the SNP
- DNA sequence spanning from the SNP to the transcription start site (TSS)
- Base-pair-resolution ATAC-seq signals for both the SNP region and the SNP-to-TSS region
- Tissue identity and auxiliary metadata (minor allele frequency, TSS distance)

**Output**: For each test sample, produce:
- A binary classification label (0 or 1) predicting the direction of gene expression change
- A continuous slope value predicting the magnitude of the eQTL effect

## 4. Evaluation Metrics

This task involves both classification and regression sub-tasks, each evaluated with its own metrics.

### Classification Metrics

#### Primary Metric
*   **AUC** (Area Under the ROC Curve): Measures the model's ability to discriminate between up-regulation and down-regulation across all classification thresholds. Higher is better. Reported per distance bin and averaged across bins.

#### Other Metrics
*   **Accuracy**: Fraction of correctly classified samples. Higher is better.
*   **F1-Score**: Harmonic mean of precision and recall for the positive class (up-regulation, label=1). Higher is better.

### Regression Metrics
*   **Pearson Correlation**: Pearson correlation coefficient between predicted and true eQTL slopes. Higher is better.
*   **RMSE**: Root mean squared error between predicted and true slopes. Lower is better.
*   **R-squared**: Coefficient of determination. Higher is better.

All metrics are computed per distance bin, then the macro-average across the four bins is also reported.
