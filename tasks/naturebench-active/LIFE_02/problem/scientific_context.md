# Lasso Peptide Property Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: cyclase_peptide_pair, ubonodin_rnap_inhibition. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Lasso peptides are a class of ribosomally synthesized and post-translationally modified peptides (RiPPs) that adopt a unique threaded lariat knot conformation. During biosynthesis, a lasso cyclase enzyme folds the core peptide into this characteristic threaded shape, which is kinetically trapped by macrolactam formation. Lasso peptides are predominantly derived from bacteria and exhibit diverse bioactivities including antibacterial, antiviral, and anticancer properties, making them promising scaffolds for drug discovery and biomedical applications.

Two fundamental challenges in lasso peptide research are (1) predicting whether a given cyclase enzyme is compatible with a given peptide substrate, and (2) predicting the biological activity of peptide variants. Cyclase-substrate compatibility is governed by complex and poorly understood sequence-function relationships, and experimental profiling of substrate tolerance is resource-intensive. Similarly, predicting the bioactivity (such as RNA polymerase inhibitory activity) of peptide variants from sequence alone is difficult due to the complexity of sequence-activity landscapes and limited experimental data. Accurate computational prediction of these properties would accelerate the rational design and discovery of lasso peptides with tailored functions.

## 2. Task Objective

This task involves two prediction sub-tasks in the lasso peptide domain:

**Instance 1: Cyclase-Peptide Pair Compatibility (cyclase_peptide_pair)**
*   **Input**: A pair of amino acid sequences -- a lasso cyclase enzyme sequence and a lasso peptide core sequence.
*   **Output**: A binary prediction of whether the cyclase-peptide pair is compatible (1) or incompatible (0).

**Instance 2: RNA Polymerase Inhibition Prediction (ubonodin_rnap_inhibition)**
*   **Input**: A 28-residue amino acid sequence of a ubonodin peptide variant.
*   **Output**: A continuous enrichment value predicting the variant's RNA polymerase (RNAP) inhibitory activity.

Both instances provide RODEO-predicted lasso precursor peptide sequences that may be used for domain-adaptive pretraining of protein language models.

## 4. Evaluation Metrics

### cyclase_peptide_pair Instance

#### Primary Metric
*   **Accuracy**: The fraction of correctly classified cyclase-peptide pairs. Higher is better.

### ubonodin_rnap_inhibition Instance

#### Primary Metric
*   **Pearson Correlation**: Pearson correlation coefficient between predicted and true enrichment values. Higher is better. Measures linear correlation.

#### Other Metrics
*   **Spearman Correlation**: Spearman rank correlation coefficient between predicted and true enrichment values. Higher is better. Measures monotonic relationship.
*   **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true enrichment values. Lower is better.
