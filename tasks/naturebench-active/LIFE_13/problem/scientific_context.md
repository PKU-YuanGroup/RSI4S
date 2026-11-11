# Inverse Protein Folding

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: cath_4_2, cath_4_3, pdb2022, ts50. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Proteins are three-dimensional structures folded from linear amino acid (AA) sequences, and they play critical roles in biological processes including metabolism, immune response, and cell cycle control. Inverse protein folding (also known as computational protein sequence design) is the problem of generating valid amino acid sequences that can fold into a desired three-dimensional backbone structure. This is a fundamental problem in computational biology and protein engineering, with applications in therapeutic protein design, lead compound optimization, and antibody engineering.

The challenge lies in accurately predicting amino acid identities from backbone geometry alone, particularly for structurally uncertain regions such as intrinsically disordered segments. Effective methods must capture both local structural constraints and long-range residue interactions to produce sequences that are not only accurate but also physically plausible, maintaining the physico-chemical and structural characteristics of native proteins.

## 2. Task Objective

Given a protein backbone structure (3D coordinates of backbone atoms), predict the amino acid sequence that would fold into that structure.

*   **Input**: Protein backbone structures represented as 3D coordinates of four backbone atoms (N, CA, C, O) for each residue, along with a protein chain identifier.
*   **Output**: A predicted single-letter amino acid sequence for each protein chain.

## 4. Evaluation Metrics

### Primary Metric

*   **recovery_rate**: The proportion of correctly predicted amino acids across all residue positions in the test set. Higher is better. Computed as the fraction of positions where the predicted amino acid exactly matches the native amino acid. Reported as a percentage (0-100%).

### Other Metrics

*   **nssr_blosum62**: Evaluates the similarity between predicted and native residues using the BLOSUM62 substitution matrix. For each residue position, a prediction is considered a positive match if the BLOSUM62 score between the predicted and native amino acid is greater than zero. The NSSR is the proportion of positive matches across all positions. Higher is better. Reported as a percentage (0-100%).
