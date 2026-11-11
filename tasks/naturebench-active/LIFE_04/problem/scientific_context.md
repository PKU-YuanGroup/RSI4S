# Cancer Gene Identification on Biological Networks

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: cpdb, iref_v15, iref_v9, ltg, mtg, multinet, pcnet, stringdb. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Identifying genes that drive cancer development is a fundamental challenge in computational biology and precision oncology. Biological networks, such as protein-protein interaction (PPI) networks and heterogeneous regulatory networks, encode relationships between genes that can reveal functional roles in tumorigenesis. By integrating network topology with multi-omics molecular profiles (somatic mutations, DNA methylation, gene expression, and copy number alterations), it becomes possible to distinguish cancer-associated genes from non-cancer genes in a data-driven manner.

This task requires developing graph-based machine learning methods that can effectively leverage both network structure and multi-omics node features to identify cancer genes across diverse biological network topologies, including both homogeneous PPI networks and heterogeneous regulatory networks with multiple node and edge types.

## 2. Task Objective

Given a biological network with multi-omics node features and partial binary labels (cancer gene vs. non-cancer gene), predict the cancer gene status for a designated set of test nodes.

*   **Input**: An attributed graph consisting of (1) an adjacency matrix encoding biological interactions, (2) a 64-dimensional multi-omics feature vector per node, (3) binary training/validation labels for a subset of nodes, and (4) a test mask identifying which nodes require prediction.
*   **Output**: A predicted probability score (between 0 and 1) for each test node, indicating the likelihood of being a cancer gene.

This is a transductive node classification task: the full graph and all node features are available during both training and inference. Only the training and validation labels should be used for model development.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: AUPRC
*   **Definition**: Area Under the Precision-Recall Curve. Measures the trade-off between precision and recall across all classification thresholds. Higher is better. Particularly informative for imbalanced datasets where the positive class is rare.

### Other Metrics
*   **AUROC**: Area Under the Receiver Operating Characteristic Curve. Measures the trade-off between true positive rate and false positive rate. Higher is better.
