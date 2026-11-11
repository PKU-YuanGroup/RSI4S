# Targeted Molecular Generation via SMILES-Based Optimization

## 1. Scientific Problem

De novo molecular design aims to computationally generate novel molecules with desired chemical properties, addressing the fundamental challenge that the space of synthetically accessible drug-like molecules is astronomically large (estimated at 10^60) while only a tiny fraction have been experimentally characterized. Deep generative models trained on SMILES (simplified molecular-input line-entry system) representations have emerged as a promising approach, where a generative model is optimized to produce molecules matching a target property profile.

A key challenge arises when the target molecular profile is under-represented or entirely absent from the training data. Standard generative methods can struggle to extrapolate beyond the training distribution, often producing molecule sets with reduced diversity or failing to satisfy complex property constraints. This task evaluates methods that can generate specific and diverse sets of molecules satisfying target property criteria, particularly when the desired properties are not well represented in the available training data.

## 2. Task Objective

Given a training corpus of drug-like SMILES molecules (with the target property profile removed or under-represented), a pre-trained RNN prior model, and a target property specification, develop a method that generates 500 novel molecules satisfying the target criterion.

*   **Input**: A filtered set of ChEMBL drug-like SMILES molecules (training data), a pre-trained RNN prior model (PyTorch checkpoint), and a target specification defining the desired molecular property (target molecule SMILES for structural similarity, or a numerical property range/threshold).
*   **Output**: A set of exactly 500 generated SMILES molecules optimized toward the specified target property.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Success Rate
*   **Definition**: The proportion of the 500 generated molecules that satisfy the target property criterion. Higher is better.
    - For `tanimoto_mol_a`: fraction of molecules with Tanimoto similarity (Morgan fingerprints, radius 2, 2048 bits) to the target molecule above 0.9.
    - For `tpsa`: fraction of molecules with TPSA (computed via RDKit) in the range [250, 300].
    - For `qed`: fraction of molecules with QED (computed via RDKit) above 0.8.

### Other Metrics
*   **Scaffold Diversity**: The number of unique Murcko scaffolds among the generated molecules that satisfy the target criterion. Higher is better. Murcko scaffolds are computed using RDKit.
