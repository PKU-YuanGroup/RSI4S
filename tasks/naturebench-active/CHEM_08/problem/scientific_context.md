# Multiproperty Molecular Optimization

## 1. Scientific Problem

Molecular optimization is a critical task in drug discovery and materials design. Given a molecule with suboptimal properties, the goal is to modify its structure to improve targeted molecular properties while maintaining sufficient structural similarity to the original molecule. This is particularly challenging when multiple properties must be optimized simultaneously, as improvements in one property may come at the cost of another.

In drug design, candidate molecules must satisfy multiple criteria related to absorption, distribution, metabolism, excretion, and toxicity (ADMET), as well as target binding affinity and drug-likeness. Efficient computational methods for multiproperty molecular optimization can dramatically accelerate the drug development pipeline by proposing structurally modified molecules that better satisfy desired property profiles.

## 2. Task Objective

Given a set of source molecules and oracle functions that evaluate molecular properties, develop a method that generates optimized molecules satisfying property constraints for each source molecule.

*   **Input**: Source molecule SMILES strings, training molecular pairs (source-target SMILES with associated property labels), and property prediction oracle models/functions.
*   **Output**: For each test source molecule, a set of optimized molecule SMILES strings. Each optimized molecule should satisfy the property constraints defined by the evaluation criteria for the given task type.

## 4. Evaluation Metrics

### drd2_qed instance

#### Primary Metric
*   **Name**: Success Rate (SR) at delta=0.4 (reported as SR@0.4)
*   **Definition**: The fraction of test molecules for which at least one of the 20 generated molecules satisfies all property thresholds (DRD2 >= 0.5, QED >= 0.6) and the similarity constraint (Tanimoto similarity >= 0.4). Higher is better.

#### Other Metrics
*   **Name**: Success Rate (SR) at delta=0.5 and delta=0.6 (reported as SR@0.5, SR@0.6)
*   **Definition**: Same as SR@0.4 but with stricter similarity thresholds (delta = 0.5 and 0.6 respectively). Higher is better.

### admet instance

#### Primary Metric
*   **Name**: Overall Success Rate of All Generated Molecules (SRA_overall)
*   **Definition**: The fraction of all generated molecules across all 10 task types that surpass the original molecule in the targeted properties, aggregated over all tasks. Higher is better.

#### Other Metrics
*   **Name**: Per-task Success Rate of All Generated Molecules (SRA per task)
*   **Definition**: For each of the 10 task types (hERG, ESOL, BBBP, Mutagenicity, lipop, hERG ESOL, Mutagenicity BBBP, hERG Mutagenicity, Mutagenicity ESOL, Mutagenicity lipop), SRA is the fraction of all generated molecules (5 per test molecule) that surpass the original molecule in the targeted properties. Property-specific thresholds: hERG/Mutag -- source prediction >= 0.5 and optimized prediction < 0.5; BBBP -- source prediction < 0.5 and optimized prediction >= 0.5; ESOL/lipop -- optimized prediction minus source prediction <= -0.5. For multiproperty tasks, both property thresholds must be satisfied simultaneously. Higher is better.
