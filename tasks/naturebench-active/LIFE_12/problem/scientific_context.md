# Enzyme Optimum pH Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: phopt. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Enzyme activity is substantially influenced by the pH of the reaction environment, often with a marked decline in activity beyond a specific pH range. The optimal pH (pHopt) at which an enzyme exhibits maximal catalytic activity is a critical property for biotechnological applications. While most characterized enzymes have a pHopt near neutral (pH 7), some function optimally at extreme acidic or alkaline conditions (as low as pH 1.0 or as high as pH 12.5).

Accurate computational prediction of enzyme pHopt from protein sequence alone would greatly accelerate enzyme discovery and engineering -- enabling high-throughput screening of candidate enzymes for industrial processes, guiding directed evolution, and informing protein design for specific pH environments. This task is challenging because the relationship between sequence and pHopt involves complex structural and biophysical determinants, including the distribution of charged residues on the protein surface, the electrostatic environment near the catalytic center, and solvent accessibility patterns.

## 2. Task Objective

Given enzyme protein sequences, predict the catalytic optimum pH (pHopt) value for each enzyme.

*   **Input**: Protein amino acid sequences (variable length, 33-1,021 residues) along with auxiliary metadata (organism name, EC number). A training set of 7,124 labeled sequences, a validation set of 760 labeled sequences, and an optional transfer learning pretraining dataset of ~1.9 million secreted bacterial proteins with environment pH labels are provided for model development.
*   **Output**: A continuous pHopt prediction (float) for each of the 1,971 test sequences.

## 4. Evaluation Metrics

All evaluation metrics are computed using **sample weights** (bin-inverse weighting) to ensure that performance on rare acidic and alkaline enzymes is represented fairly despite the imbalanced pH distribution. The sample weights are pre-computed based on three pH bins: acidic (pH <= 5), neutral (5 < pH < 9), and alkaline (pH >= 9).

### Primary Metric
*   **Name**: Sample-weighted RMSE
*   **Definition**: Root mean squared error between predicted and true pHopt values, weighted by sample weights. Lower is better.

### Other Metrics
*   **Sample-weighted R (Pearson correlation)**: Pearson correlation coefficient between predicted and true pHopt values, computed by resampling the test set according to sample weight probabilities and averaging over 100 repetitions. Higher is better.
