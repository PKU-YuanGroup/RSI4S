# Yeast Promoter Expression Prediction

## 1. Scientific Problem

Gene expression in eukaryotes is regulated by DNA sequences in promoter regions, where transcription factors bind and compete with nucleosomes to control transcription levels. Understanding the quantitative relationship between promoter DNA sequence and resulting gene expression is a fundamental challenge in genomics and has broad implications for synthetic biology, genetic engineering, and understanding complex trait genetics.

Random promoter libraries, in which millions of synthetic DNA sequences are placed in a promoter context and their expression levels are experimentally measured, provide a powerful system for training and evaluating sequence-to-expression models. These datasets capture the combinatorial complexity of cis-regulatory logic, including the effects of transcription factor binding site composition, positioning, and interactions.

## 2. Task Objective

Build a model that predicts gene expression levels from promoter DNA sequences in yeast.

*   **Input**: DNA promoter sequences (110 bp each, consisting of a 17-bp left adapter, 80-bp variable region, and 13-bp right adapter). A plasmid backbone file (`plasmid.json`) is provided for reconstructing full 150-bp sequence context. Training data includes 6.7 million sequences with measured expression values.
*   **Output**: Predicted expression values (continuous float) for each of the 71,103 test sequences.

## 4. Evaluation Metrics

The test set comprises eight subsets, each with a specific weight. Two composite scores are computed as weighted averages across subsets.

Subset weights: all (1.0), high_exp (0.3), low_exp (0.3), yeast (0.3), random (0.3), challenging (0.5), SNVs (1.25), motif_perturbation (0.3), motif_tiling (0.4). Total weight: 4.65.

For single-index subsets (all, high_exp, low_exp, yeast, random, challenging), Pearson r and Spearman rho are computed between predicted and measured expression values. For paired subsets (SNVs, motif_perturbation, motif_tiling), the correlation is computed on expression *differences* between paired sequences (alt minus ref).

Only the private portion (~87%) of the test set is used for evaluation; a 13% public leaderboard portion is excluded.

### Primary Metric
*   **Name**: Pearson Score
*   **Definition**: Weighted sum of squared Pearson correlation coefficients (r^2) across test subsets, divided by the sum of weights. Higher is better.

### Other Metrics
*   **Spearman Score**: Weighted sum of Spearman rank correlation coefficients (rho) across test subsets, divided by the sum of weights. Higher is better.
