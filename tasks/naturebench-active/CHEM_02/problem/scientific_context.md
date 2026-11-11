# Molecular Property Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: bace, bbbp, clintox, esol, freesolv, hiv, lipo, muv, qm7, qm8, qm9, sider, tox21. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Predicting molecular properties from molecular structure is a fundamental challenge in computational chemistry and drug discovery. Molecular properties -- ranging from biological activities (e.g., toxicity, enzyme inhibition, membrane permeability) to physicochemical quantities (e.g., solubility, lipophilicity) and quantum-mechanical observables (e.g., orbital energies, atomization energies) -- are critical for understanding molecular behavior and guiding the design of new compounds. However, experimental measurement of these properties is often costly and time-consuming, making accurate computational prediction highly valuable.

A key challenge is that labeled molecular data is scarce relative to the vast chemical space of potential compounds (estimated at 10^60 molecules). Models must learn representations that generalize across diverse molecular structures with limited supervision. This motivates approaches that leverage large unlabeled molecular databases for pre-training before fine-tuning on specific property prediction tasks.

## 2. Task Objective

Given molecular structures represented as SMILES strings, predict molecular properties for 13 benchmark datasets spanning both classification and regression tasks.

*   **Input**: Molecular SMILES strings and, where available, auxiliary molecular descriptors. A large corpus of ~10 million unlabeled SMILES strings is provided for optional unsupervised/self-supervised pre-training.
*   **Output**: For classification instances, predict the probability of the positive class for each molecule and target. For regression instances, predict continuous property values for each molecule and target.

## 4. Evaluation Metrics

### Primary Metric

Classification and regression instances use different primary metrics:

*   **ROC-AUC (%)**: For all classification instances (BACE, BBBP, ClinTox, HIV, MUV, SIDER, Tox21). Area under the receiver operating characteristic curve, reported as a percentage. Higher is better. For multi-task classification, the mean ROC-AUC across all tasks is reported.

### Other Metrics

*   **RMSE**: For regression instances FreeSolv, ESOL, and Lipo. Root mean squared error between predicted and ground truth values. Lower is better.
*   **MAE**: For regression instances QM7, QM8, and QM9. Mean absolute error between predicted and ground truth values. Lower is better. For QM8 (12 targets) and QM9 (8 targets), the mean MAE across all targets is reported.
