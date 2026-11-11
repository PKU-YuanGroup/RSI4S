# Molecular Energy and Force Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: 3bpa_1200k, 3bpa_300k, 3bpa_600k, 3bpa_dihedral. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Machine learning interatomic potentials (MLIPs) aim to approximate quantum mechanical calculations for predicting molecular energies and forces at a fraction of the computational cost. These potentials are critical for molecular dynamics simulations, enabling the study of chemical reactions, drug-protein interactions, and materials properties at scales inaccessible to direct quantum mechanical methods.

A key challenge is building models that not only achieve high accuracy on configurations similar to the training data (in-domain) but also extrapolate reliably to configurations sampled at higher temperatures or along unexplored degrees of freedom (out-of-domain). This task evaluates the ability of machine learning models to predict total energies and per-atom forces for a flexible drug-like organic molecule across multiple thermodynamic regimes.

## 2. Task Objective

Train a machine learning interatomic potential on molecular configurations at 300K, then predict total energies and per-atom forces for test configurations across four evaluation regimes.

*   **Input**: Molecular configurations in extended XYZ format, each containing 27 atoms (C12H12N2O) with atomic species and Cartesian positions. Training data includes DFT-computed energies (eV) and forces (eV/Angstrom). Isolated atom reference energies are provided.
*   **Output**: For each test configuration, predict the total energy (eV) and per-atom forces (eV/Angstrom, shape [27, 3]).

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Energy RMSE (meV)
*   **Definition**: Root-mean-square error of predicted total energies relative to DFT reference energies, reported in millielectronvolts (meV). Lower is better.

    `Energy RMSE = sqrt(mean((E_pred - E_ref)^2)) * 1000` (converting eV to meV)

### Other Metrics
*   **Force RMSE (meV/A)**: Root-mean-square error of predicted per-atom force components relative to DFT reference forces, reported in meV/Angstrom. The RMSE is computed over all atoms and all three Cartesian components across all test configurations. Lower is better.

    `Force RMSE = sqrt(mean((F_pred - F_ref)^2)) * 1000` (converting eV/A to meV/A)

    where the mean is taken over all configurations, all atoms, and all three force components.
