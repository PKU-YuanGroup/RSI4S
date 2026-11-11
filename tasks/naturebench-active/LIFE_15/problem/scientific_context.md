# Protein Essentiality Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: cell_kyse70, human, mouse. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Essential proteins are indispensable for an organism's survival, performing core biological functions related to growth, development, and cellular maintenance. Identifying which proteins are essential is crucial for drug target discovery, therapeutic development, and synthetic biology applications. Experimental methods such as gene knockout, RNA interference, and CRISPR screening can identify essential proteins but are costly, time-consuming, and context-dependent.

Protein essentiality is not absolute -- it varies depending on the biological context. A protein may be essential at the organism level (intolerant to loss-of-function variants in human populations), at the animal model level (causing embryonic lethality when knocked out in mice), or at the cell-line level (required for cell viability in specific cancer cell lines). Computational methods that can accurately predict protein essentiality from sequence alone, across these different biological contexts, would greatly accelerate biomedical research.

## 2. Task Objective

Given protein amino acid sequences, predict binary essentiality labels across three biological levels.

*   **Input**: Protein amino acid sequences (variable-length strings of amino acid characters).
*   **Output**: Binary predictions (0 = non-essential, 1 = essential) for each test protein, with the definition of essentiality varying by instance.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: AUROC (Area Under the Receiver Operating Characteristic Curve)
*   **Definition**: Measures the model's ability to discriminate between essential and non-essential proteins across all classification thresholds. Higher is better. Range: [0, 1].

### Other Metrics
*   **AUPRC** (Area Under the Precision-Recall Curve): Measures precision-recall trade-off, particularly informative for imbalanced datasets. Higher is better.
*   **Accuracy**: Fraction of correctly classified samples (using threshold 0.5). Higher is better.
*   **Precision**: Fraction of predicted positives that are true positives (using threshold 0.5). Higher is better.
*   **Recall**: Fraction of true positives that are correctly predicted (using threshold 0.5). Higher is better.
*   **F1-Score**: Harmonic mean of precision and recall (using threshold 0.5). Higher is better.
