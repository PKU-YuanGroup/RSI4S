# Molecular Interactions and Properties Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: bace, bbbp, bindingdb, esol, freesolv, lipophilicity, sider, tox21, toxcast. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Drug discovery is a lengthy and costly process that relies heavily on understanding molecular interactions and properties. Predicting how molecules interact with biological targets (proteins, other drugs) and what physical-chemical, pharmacokinetic, or toxicological properties they exhibit is fundamental to identifying viable drug candidates. Accurate computational predictions can dramatically accelerate this process by narrowing the vast chemical space to the most promising candidates before expensive experimental validation.

This task spans multiple facets of molecular prediction: predicting physical-chemistry properties (water solubility, lipophilicity, hydration free energy), predicting bioactivity and toxicity profiles (beta-secretase inhibition, blood-brain barrier penetration, side effects, toxicity across multiple assay panels), and predicting protein-ligand binding interactions. Each of these represents a critical checkpoint in the drug development pipeline.

## 2. Task Objective

Build predictive models for molecular properties and molecular interactions across nine benchmark datasets.

*   **Input**: Molecular SMILES strings (for property prediction) or compound SMILES + protein amino acid sequences (for protein-ligand binding prediction). Detailed input formats are specified per instance in the data.
*   **Output**: Predicted property values (continuous, for regression instances) or predicted probability scores (float in [0, 1], for classification instances). See Section 5 for the exact output format.

## 4. Evaluation Metrics

### Regression Instances (esol, lipophilicity, freesolv)

#### Primary Metric
*   **Name**: RMSE
*   **Definition**: Root Mean Squared Error between predicted and true property values. Lower is better.

### Classification Instances (bace, bbbp, sider, tox21, toxcast, bindingdb)

#### Primary Metric
*   **Name**: AUC
*   **Definition**: Area Under the Receiver Operating Characteristic Curve, measuring discrimination ability between positive and negative classes. Higher is better. For multi-task instances (sider, tox21, toxcast), AUC is computed per task and then averaged across tasks, skipping tasks where ground truth contains only one class or all values are missing.
