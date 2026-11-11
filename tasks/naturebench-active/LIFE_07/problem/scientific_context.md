# De Novo Nucleic Acid Atomic Structure Modeling from Cryo-EM Maps

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: benchmark, sars_cov2. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Determining the three-dimensional atomic structures of nucleic acids (DNA and RNA) is fundamental to understanding their biological functions, including roles in gene regulation, translation, and viral replication. Cryogenic electron microscopy (cryo-EM) has become a primary technique for resolving biomolecular structures, producing three-dimensional electron density maps at resolutions ranging from near-atomic to medium resolution (2--5 Angstroms). However, building accurate atomic-level structure models from these density maps remains a significant challenge, particularly for nucleic acids, which exhibit greater structural flexibility and more diverse local patterns than proteins.

Automated de novo structure modeling methods are needed to convert cryo-EM density maps into full atomic coordinate models of nucleic acids. This task is especially difficult at medium resolutions (3--5 Angstroms), where individual atoms are not directly resolvable and the distinction between different nucleotide types becomes ambiguous. An effective method must identify the positions of backbone components (phosphate and sugar moieties), trace the backbone connectivity, assign the correct nucleotide sequence, and produce a complete atomic model.

## 2. Task Objective

Given a cryo-EM electron density map of a biomolecular complex containing nucleic acids, build a de novo atomic structure model of the DNA/RNA components.

*   **Input**: A cryo-EM density map (gzip-compressed MRC format), the nucleic acid sequences present in the map (FASTA format), and map metadata (resolution in Angstroms). Training data consisting of 290 cryo-EM maps with corresponding reference atomic structures is provided for model development.
*   **Output**: A predicted atomic structure model of the nucleic acid components in PDB format, containing three-dimensional coordinates for all modeled nucleotide atoms (backbone: phosphate and sugar atoms; bases: nucleotide-specific atoms).

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: backbone_recall
*   **Definition**: The fraction of backbone atoms (sugar and phosphate moieties) in the reference structure that are correctly modeled. For each nucleotide in the reference, the sugar atoms (C1', C2', O2' for RNA, C3', O3', C4', O4', C5') and phosphate atoms (P, OP1, OP2, O5', OP3') are checked against the predicted model. A nucleotide's backbone is considered correctly modeled if the average distance between matched atoms is within 5 Angstroms. Higher is better. Reported as a value in [0, 1], averaged over all nucleotides in the reference structure.

### Other Metrics
*   **sequence_recall**: The fraction of nucleotides in the reference structure whose base type is correctly identified in the predicted model. A nucleotide is first matched to its nearest counterpart in the prediction (by average atom distance), then the base types (A, C, G, U/T) are compared. Higher is better. Reported as a value in [0, 1].
