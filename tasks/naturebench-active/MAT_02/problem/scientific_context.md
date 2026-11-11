# Metal-Organic Framework Property Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: band_gap, h2_diffusivity, h2_uptake. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Metal-organic frameworks (MOFs) are crystalline porous materials composed of metal nodes connected by organic linkers, forming highly tunable three-dimensional networks. Their structural diversity -- arising from the vast combinatorial space of building blocks and topologies -- makes them promising candidates for applications in gas storage, separation, catalysis, and electronics. However, experimentally characterizing every possible MOF is infeasible, motivating computational approaches to predict MOF properties from their structural features.

MOF properties span fundamentally different physical domains: gas adsorption depends primarily on pore geometry and topology (global features), gas diffusion is influenced by both geometric and chemical factors, and electronic properties like band gaps depend strongly on local atomic chemistry. Developing a unified predictive framework that captures both local chemical features (individual atoms, bonds, functional groups) and global geometric/topological features (pore structure, framework topology, energy landscapes) is a key challenge in computational materials science.

## 2. Task Objective

Given pre-computed multi-modal embeddings of MOF structures (atom-based graph embeddings capturing local features, and energy-grid embeddings capturing global features), predict continuous physical or electronic property values for each structure.

*   **Input**: For each MOF structure, three pre-computed embedding files are provided:
    - `.graphdata`: Atom-based graph embedding (atomic numbers, neighbor indices, distances, unique atom groups) -- captures local structural and chemical features
    - `.grid` + `.griddata16`: Energy grid metadata and data (3D van der Waals interaction energy field) -- captures global geometric and energetic features
*   **Output**: A single floating-point property value per structure, specific to each instance:
    - `h2_uptake`: H2 uptake (cm^3/cm^3) at 100 bar and 77 K
    - `h2_diffusivity`: Log-transformed H2 diffusivity (log(cm^2/s)) at infinite dilution and 77 K
    - `band_gap`: PBE band gap energy (eV)

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: R2
*   **Definition**: Coefficient of determination (R-squared). Measures the proportion of variance in the target property explained by predictions. Higher is better. R2 = 1 - SS_res / SS_tot, where SS_res = sum((y_true - y_pred)^2) and SS_tot = sum((y_true - mean(y_true))^2).

### Other Metrics
*   **MAE**: Mean absolute error. MAE = mean(|y_true - y_pred|). Lower is better. Units match the target property (cm^3/cm^3 for h2_uptake, log(cm^2/s) for h2_diffusivity, eV for band_gap).
