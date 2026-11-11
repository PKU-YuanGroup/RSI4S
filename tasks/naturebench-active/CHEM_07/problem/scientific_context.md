# Molecular Linker Design

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: casf, geom, pockets, zinc. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Fragment-based drug discovery is an effective paradigm in early-stage drug development. Molecular fragments that bind to target protein sites are identified through experimental screening, and these fragments must then be connected by chemical linkers to form complete drug-like molecules. Designing linkers that produce valid, synthetically accessible, and drug-like molecules while respecting the 3D geometry of the input fragments is a challenging generative chemistry problem.

The difficulty varies with the number of fragments (pairs vs. multiple fragments) and the availability of structural context (isolated fragments vs. protein pocket environment). A successful linker design method must generate molecules that are chemically valid, synthetically accessible, drug-like, and geometrically consistent with the reference structures.

## 2. Task Objective

Given a set of disconnected 3D molecular fragments (represented as atomic point clouds with atom types, coordinates, and anchor points), generate linker atoms that connect them into a complete, valid molecule.

*   **Input**: 3D atomic point clouds of molecular fragments. Each test example provides fragment atom positions, one-hot atom types, charges, and anchor atom indices. Some instances additionally provide protein pocket atom context.
*   **Output**: For each test example, generate multiple candidate linked molecules. Output both SMILES representations and 3D molecular structures (SDF format).

## 4. Evaluation Metrics

### Primary Metric
*   **Validity** (%): Percentage of generated molecules that are chemically valid (pass RDKit sanitization and contain all atoms from the input fragments). Higher is better.

### Other Metrics
*   **QED** (0-1): Mean quantitative estimate of drug-likeness of valid generated molecules. Higher is better.
*   **SA** (1-10): Mean synthetic accessibility score of valid generated molecules. Lower is better (more accessible).
*   **Uniqueness** (%): Percentage of unique molecules among valid generations per test example. Higher is better.
*   **Novelty** (%): Percentage of valid generated linkers not seen in the training set. Higher is better.
*   **Recovery** (%): Percentage of test examples for which at least one generated molecule exactly matches the ground truth (by canonical SMILES, ignoring stereochemistry). Higher is better.
*   **RMSD** (Angstroms): Mean root-mean-square deviation between generated and ground-truth linker atom coordinates, computed only for recovered molecules. Lower is better.
*   **SCRDKit** (0-1): Mean shape-chemical similarity score between generated and ground-truth molecules, combining shape overlap and pharmacophoric feature matching. Computed for all valid molecules. Higher is better.
