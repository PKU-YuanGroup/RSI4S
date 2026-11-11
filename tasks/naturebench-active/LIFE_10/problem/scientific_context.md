# Protein Variant Effect Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: dlg4_abundance, dlg4_binding, gb1, gfp, grb2_abundance, grb2_binding, pab1, pten_abundance, pten_activity, tem_1, ube4b. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Proteins are the molecular machines of life, and engineering them with improved or novel properties is a central challenge in biotechnology and biomedical research. Deep mutational scanning (DMS) experiments systematically measure the functional effects of thousands of amino acid substitutions in a protein, generating sequence-function landscapes. However, these experiments are expensive and cannot exhaustively cover the vast combinatorial space of possible mutations. Computational methods that can accurately predict the functional effects of protein variants from sequence information are therefore critically needed to guide protein engineering efforts.

The key scientific challenge is to learn the mapping from protein sequence variants to their experimentally measured functional properties. This mapping is complex because protein function depends on the interplay between sequence, three-dimensional structure, and biophysical properties. Models must generalize from observed training variants to predict the effects of unseen mutations, including mutations at positions not represented in the training data and combinations of mutations not previously tested.

## 2. Task Objective

Given a set of protein sequence variants (defined by their amino acid substitutions relative to a wild-type reference) and corresponding experimentally measured functional scores as training data, predict the functional scores for a held-out set of test variants.

*   **Input**: For each instance, a training set of protein variants with their amino acid substitutions, mutation counts, and experimentally measured functional scores (tab-separated values). A held-out test set of protein variants with their amino acid substitutions and mutation counts (no scores). A PDB file containing the three-dimensional structure of the wild-type protein.
*   **Output**: For each test variant, a predicted functional score (continuous real number).

## 4. Evaluation Metrics

### Primary Metric

*   **Name**: Spearman Correlation
*   **Definition**: Spearman's rank correlation coefficient between predicted scores and experimentally measured ground-truth scores. Computed as the Pearson correlation of the rank-transformed predicted and reference values. Ranges from -1 to 1. Higher is better.
