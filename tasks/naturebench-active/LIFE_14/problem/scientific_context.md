# Protein Stability Change Prediction Upon Mutations

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: s571, s669. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Protein stability is a fundamental property in protein science and engineering, commonly measured by the unfolding free energy change (DDG, in kcal/mol) and the melting temperature change (DTm, in degrees Celsius) upon mutation. Accurate computational prediction of how single-point mutations affect protein stability is critical for rational protein design, understanding disease-associated variants, and guiding biotechnological applications. However, the limited availability of experimentally measured stability data and the complexity of protein structure-function relationships make this a challenging prediction task.

This task evaluates the ability of machine learning methods to predict thermodynamic stability changes of proteins upon single-point amino acid mutations, given protein sequences, mutation information, and experimental conditions.

## 2. Task Objective

Given a protein mutation (wild-type and mutant sequences) along with experimental conditions, predict the resulting change in protein stability.

*   **Input**: Wild-type protein sequence, mutant protein sequence (differing at exactly one residue), experimental pH, and (for DDG prediction only) experimental temperature.
*   **Output**: A numerical prediction of the stability change for each mutation: DDG in kcal/mol for the `s669` instance, or DTm in degrees Celsius for the `s571` instance.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Spearman Correlation Coefficient
*   **Definition**: Spearman rank correlation between predicted and experimental stability change values. Higher is better. Measures how well predictions preserve the ranking of mutation effects.

### Other Metrics
*   **Pearson Correlation Coefficient**: Linear correlation between predicted and experimental values. Higher is better.
*   **RMSE**: Root mean squared error between predicted and experimental values. Lower is better.
*   **MAE**: Mean absolute error between predicted and experimental values. Lower is better.
