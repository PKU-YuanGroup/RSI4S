# Transition State Structure Generation

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: transition1x. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Transition states (TSs) are transient molecular configurations that occur at the highest-energy point along a reaction pathway between reactants and products. Accurately predicting TS three-dimensional structures is critical for understanding reaction mechanisms, estimating reaction rates, and designing catalysts. However, because TSs are first-order saddle points on the potential energy surface, they are extremely difficult to characterize experimentally and computationally expensive to locate using traditional quantum chemistry methods such as nudged elastic band (NEB) coupled with density functional theory (DFT).

Machine learning approaches to TS structure generation aim to bypass the costly quantum chemistry optimization by directly predicting the 3D atomic coordinates of the transition state given the 3D structures of the reactants and products. This "double-ended" formulation treats TS generation as a conditional 3D structure prediction task, where the input is a pair of molecular conformations (reactant and product) and the output is the corresponding TS conformation.

## 2. Task Objective

Given paired reactant and product 3D molecular structures for a set of elementary organic reactions, predict the transition state 3D structure for each reaction.

*   **Input**: For each test reaction, the 3D atomic coordinates and atomic numbers of the reactant and product molecules.
*   **Output**: For each test reaction, the predicted 3D atomic coordinates of the transition state structure.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: mean_rmsd
*   **Definition**: Mean root mean square deviation (RMSD) in Angstroms between predicted and reference transition state structures across all test reactions. RMSD is computed after optimal Kabsch alignment, with reflection considered (the minimum of the direct and reflected RMSD is taken). Lower is better.

### Other Metrics
*   **median_rmsd**: Median RMSD in Angstroms across all test reactions. Lower is better.
