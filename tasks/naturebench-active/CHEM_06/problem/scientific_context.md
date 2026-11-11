# Pocket-Conditioned 3D Molecular Generation

## 1. Scientific Problem

Structure-based drug design aims to discover molecules that bind strongly to specific protein targets. A key challenge is generating novel drug-like molecules that are geometrically compatible with protein binding pockets -- the three-dimensional cavities on a protein surface where small molecules can bind and modulate biological function.

Generating molecules conditioned on pocket structure requires models that understand both the chemical space of valid drug-like molecules and the geometric constraints imposed by the target protein's binding site. Generated molecules should not only possess favorable drug-likeness properties (such as solubility, synthesizability, and adherence to medicinal chemistry rules) but should also be structurally diverse to provide a rich set of candidates for downstream optimization.

## 2. Task Objective

Train a generative model on protein-ligand complex data, then generate novel 3D molecules for unseen test protein binding pockets.

*   **Input**: For each test pocket, a set of residue-level features describing the 3D structure of the protein binding pocket (C-alpha coordinates, backbone geometry, amino acid types) and the pocket center coordinates. Training data consists of approximately 100,000 protein-ligand pairs with full atomic detail.
*   **Output**: For each test pocket, a set of generated molecules represented as SMILES strings with associated drug-likeness properties.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Vina Score
*   **Definition**: Binding affinity estimated by molecular docking using QuickVina2. Reported in kcal/mol. More negative values indicate stronger predicted binding to the target pocket. Lower (more negative) is better. Evaluated by generating a 3D conformer from the output SMILES, then docking against the receptor pocket structure.

### Other Metrics
*   **QED**: Quantitative Estimation of Drug-likeness. A composite score in [0, 1] integrating molecular weight, lipophilicity, polar surface area, hydrogen bond donors/acceptors, aromaticity, and structural alerts. Higher is better.
*   **SA**: Normalized Synthetic Accessibility score. Computed as (10 - raw_SA) / 9, where raw_SA is the RDKit SA score (1-10). The normalized score ranges from 0 (difficult to synthesize) to 1 (easy to synthesize). Higher is better.
*   **Diversity**: Internal diversity of the generated molecule set, computed as 1 minus the average pairwise Tanimoto similarity of Morgan fingerprints. Higher is better.
