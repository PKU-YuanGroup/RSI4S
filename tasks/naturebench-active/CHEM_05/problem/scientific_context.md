# Molecular and Protein Property Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: buchwald_hartwig, esol, fluorescence, freesolv, lipophilicity, stability, suzuki. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Predicting physicochemical and biophysical properties from molecular and protein sequences is a foundational challenge in computational chemistry and biology. Accurate property prediction enables virtual screening of drug candidates, understanding protein function, and optimizing chemical reactions -- reducing the need for expensive and time-consuming laboratory experiments.

This task spans three property prediction domains: (1) predicting physicochemical properties of small molecules from their chemical structure (water solubility, hydration free energy, lipophilicity), (2) predicting biophysical properties of proteins from their amino acid sequences (fluorescence intensity, folding stability), and (3) predicting reaction yields from reaction components or reaction SMILES. Each domain presents unique challenges in learning structure-property relationships from limited labeled data.

## 2. Task Objective

Given molecular structures (as SMILES strings), protein sequences (as amino acid strings), or chemical reaction representations, predict their associated continuous property values.

*   **Input**: Molecular SMILES strings, protein amino acid sequences, or reaction SMILES/component representations, depending on the instance.
*   **Output**: A continuous numerical prediction for each test sample (e.g., solubility value, fluorescence intensity, reaction yield).

## 4. Evaluation Metrics

Different metrics are used for different instance groups, reflecting community conventions in each domain.

### Small-Molecule Instances (esol, freesolv, lipophilicity)
*   **RMSE** (Root Mean Square Error): Lower is better. Computed as sqrt(mean((y_pred - y_true)^2)). This is the primary metric for these instances.

### Protein Instances (fluorescence, stability)
*   **Spearman's rho** (Spearman Rank Correlation): Higher is better. Measures the monotonic relationship between predicted and true values using rank correlation. This is the primary metric for these instances.

### Reaction Yield Instances (buchwald_hartwig, suzuki)
*   **R-squared** (Coefficient of Determination): Higher is better. Computed as 1 - SS_res/SS_tot, where SS_res = sum((y_true - y_pred)^2) and SS_tot = sum((y_true - mean(y_true))^2). For multi-split instances, R-squared is computed per split and averaged. This is the primary metric for these instances.
