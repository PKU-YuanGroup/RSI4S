# Protein-Protein Interaction Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: bernett, cross_species_ecoli, cross_species_fly, cross_species_mouse, cross_species_worm, cross_species_yeast, mutation_effect, virus_human. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Protein-protein interactions (PPIs) mediate virtually all biological processes in cells, from signal transduction to immune response. Identifying which proteins interact is critical for understanding cellular function, disease mechanisms, and host-pathogen relationships. Experimental methods for detecting PPIs are costly and time-consuming, leaving interaction databases sparse for most species. Computational prediction of PPIs from amino acid sequences alone offers a scalable alternative, but remains challenging due to the complexity of interaction patterns and the difficulty of generalizing across species.

This task encompasses four related PPI prediction challenges: cross-species generalization (training on human PPIs and predicting interactions in other organisms), leakage-controlled human PPI prediction, mutation effect classification (predicting whether mutations increase or decrease interaction strength), and virus-human interaction prediction.

## 2. Task Objective

Develop a method to predict protein-protein interactions from amino acid sequences.

*   **Input**: Pairs (or triplets) of protein amino acid sequences. For standard PPI prediction, each sample consists of two protein sequences. For mutation effect prediction, each sample consists of a wild-type protein sequence, a mutant protein sequence, and an interaction partner sequence.
*   **Output**: Binary predictions indicating whether proteins interact (PPI instances) or whether a mutation increases interaction strength (mutation effect instance).

## 4. Evaluation Metrics

### Primary Metric
*   **AUPR (Area Under the Precision-Recall Curve)**: Measures the trade-off between precision and recall across all classification thresholds. Higher is better. Applies to all instances.

### Other Metrics
*   **AUROC (Area Under the Receiver Operating Characteristic Curve)**: Measures discrimination ability across all thresholds. Higher is better. Applies to the `mutation_effect` instance only.
