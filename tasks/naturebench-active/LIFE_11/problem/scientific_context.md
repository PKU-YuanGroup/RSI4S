# RNA Sequence Analysis

## 1. Scientific Problem

RNA molecules play critical roles in cellular processes including gene expression, regulation, and catalysis. Computational analysis of RNA sequences is essential for understanding their functions and roles in biological systems. Two fundamental tasks in RNA analysis are: (1) classifying non-coding RNA (ncRNA) sequences into functional categories, and (2) predicting RNA secondary structures, which describe the base-pairing arrangements within an RNA sequence.

Non-coding RNA classification helps identify RNA function and ontology, which is important for understanding gene regulation and developing therapies for RNA-related diseases. RNA secondary structure prediction determines the probable arrangement of base pairs within an RNA molecule that folds back onto itself, providing critical insights into its interactions with other molecules and functional importance.

## 2. Task Objective

This benchmark consists of two distinct RNA analysis tasks:

**Task A: ncRNA Sequence Classification**
*   **Input**: RNA nucleotide sequences (characters A, U, C, G)
*   **Output**: Predicted ncRNA class label for each sequence (one of 13 categories)

**Task B: RNA Secondary Structure Prediction**
*   **Input**: RNA nucleotide sequences (characters A, U, C, G)
*   **Output**: Predicted secondary structure as base-pair annotations for each sequence, specifying which nucleotide positions are paired

## 4. Evaluation Metrics

### Classification Metrics (Instance: `nrc`)

#### Primary Metric
*   **F1-Score**: Harmonic mean of precision and recall, macro-averaged across all 13 classes. Higher is better.

#### Other Metrics
*   **Accuracy**: Fraction of correctly classified sequences. Higher is better.
*   **Precision**: Macro-averaged precision across all classes. Higher is better.
*   **Recall**: Macro-averaged recall across all classes. Higher is better.
*   **MCC (Matthews Correlation Coefficient)**: Correlation between predicted and actual classes, accounting for class imbalance. Range [-1, 1]. Higher is better.

### Structure Prediction Metrics (Instances: `archiveii600`, `ts0`)

#### Primary Metric
*   **F1-Score**: Harmonic mean of precision and recall for base pair prediction, averaged across all test structures. Higher is better.

#### Other Metrics
*   **Precision**: Fraction of predicted base pairs that are correct, averaged across all test structures. Higher is better.
*   **Recall**: Fraction of true base pairs that are correctly predicted, averaged across all test structures. Higher is better.
