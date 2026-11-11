# Protein-Protein Interaction Interface Residue Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: ppi_interface_benchmark. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Proteins interact with one another through specific binding interfaces, and identifying which residues participate in these interfaces is essential for understanding protein function, disease mechanisms, and drug design. However, only a small fraction of known protein-protein interactions have experimentally determined co-crystal structures, leaving the vast majority without structural characterization of their binding sites.

The challenge is to predict, for each residue of each protein in an interacting pair, whether that residue participates in the interaction interface. Crucially, interface predictions must be partner-specific: the same protein may use different sets of interface residues when binding to different partners. Accurate per-residue, partner-specific interface prediction at proteome scale would enable systematic analysis of how mutations disrupt specific interactions and contribute to disease.

## 2. Task Objective

Given a pair of interacting proteins, predict for each residue of each protein whether it is an interface residue (binary classification at the residue level).

*   **Input**: Pairs of interacting proteins, each identified by an opaque task-private `pair_id` and the amino-acid sequences of the two proteins. External sequence databases (e.g., UniProt) may be consulted for general feature computation, BUT the original protein identifiers for the test set have been deliberately withheld to prevent lookup of known interfaces.
*   **Output**: For each protein pair, per-residue interface probability scores for both proteins. Each score should be a floating-point value in [0, 1], where higher values indicate higher confidence that the residue is an interface residue.

## 4. Evaluation Metrics

### Primary Metric
*   **AUROC** (Area Under the Receiver Operating Characteristic Curve): Measures discriminative performance across all classification thresholds. Higher is better. Computed across all valid residues (non-nan labels) from all test interactions pooled together.

### Other Metrics
*   **AUPR** (Area Under the Precision-Recall Curve): Measures performance under class imbalance. Higher is better.
*   **F1-score**: Harmonic mean of precision and recall at threshold 0.5. Higher is better.
*   **MCC** (Matthews Correlation Coefficient): Balanced measure accounting for all four confusion matrix categories. Higher is better. Range [-1, 1].
*   **Precision**: Fraction of predicted interface residues that are true interface residues (at threshold 0.5). Higher is better.
*   **Recall**: Fraction of true interface residues that are correctly predicted (at threshold 0.5). Higher is better.
*   **Accuracy**: Fraction of correctly classified residues (at threshold 0.5). Higher is better.
