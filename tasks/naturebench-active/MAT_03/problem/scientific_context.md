# Zeolite-Molecule Binding Affinity Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: zeolite_osda_binding. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Zeolites are microporous crystalline materials widely used in industrial catalysis and separations. Their synthesis typically requires organic structure-directing agents (OSDAs) that template specific framework topologies. Understanding which molecules bind favorably to which zeolite frameworks is critical for rational OSDA design, but the enormous size of the combined molecular and framework chemical space makes exhaustive experimental or computational screening infeasible.

Predicting binding affinities between molecules and zeolite frameworks from pre-computed physical descriptors enables efficient virtual screening of large molecular libraries. This task captures three complementary aspects of binding affinity: whether a molecule fits inside a framework's cavities (binary classification), the strength of the framework-molecule interaction (energy regression), and the molecular loading per unit cell (multiclass classification). Accurate prediction across all three sub-tasks provides a comprehensive characterization of zeolite-molecule compatibility.

## 2. Task Objective

Given pre-computed physical descriptors of molecules and zeolite frameworks, predict binding affinity properties for framework-molecule pairs.

*   **Input**: Each test sample is a (SMILES, Zeolite) pair. For each molecule, a 1,667-dimensional descriptor vector is available in `osda_priors_0.pkl` (indexed by SMILES). For each framework, a 1,948-dimensional descriptor vector is available in `zeolite_priors_0.pkl` (indexed by framework code). Additional pair-level metadata is provided in `x_test.csv`.
*   **Output**: For each test pair, predict:
    1. `b_pred`: Binary binding label (0 = non-binding, 1 = binding)
    2. `E_pred`: Binding energy in kJ/mol/Si (float; 0.0 for non-binding pairs, negative for binding)
    3. `BE_pred`: Binding energy combining classification and regression (float; typically `b_pred * E_pred`)
    4. `load_class_pred`: Discrete loading classification index (integer, 0-47)

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: BE_RMSE
*   **Definition**: Root Mean Squared Error of the predicted binding energy (BE) across all test pairs, in kJ/mol/Si. Lower is better.

### Other Metrics
*   **b_F1**: F1 score for binary binding classification. Higher is better. Computed using `sklearn.metrics.f1_score` with default (binary) averaging.
*   **E_RMSE**: Root Mean Squared Error of the predicted energy (E) computed only on binding pairs (where ground truth `b=1`), in kJ/mol/Si. Lower is better.
*   **Spearman_mean**: Mean per-molecule Spearman's rank correlation coefficient between predicted and true binding energies (BE) across frameworks. Computed by pivoting predictions and ground truth into molecule-by-framework matrices, filtering to molecules and frameworks with sufficient data, then averaging the per-molecule Spearman correlations. Higher is better.
