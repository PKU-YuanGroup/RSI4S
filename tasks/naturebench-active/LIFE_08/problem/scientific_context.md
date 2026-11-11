# Protein-DNA Binding Specificity Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: benchmark. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Understanding how proteins recognize and bind specific DNA sequences is fundamental to gene regulation, with broad implications for synthetic biology and therapeutic design. A protein that binds DNA typically contacts a short stretch of the double helix, exhibiting a preference for certain nucleotide identities at each position. This preference is commonly encoded as a position weight matrix (PWM), where each row represents a binding position and each column represents the probability of a particular nucleotide. Accurately predicting this binding preference from the three-dimensional structure of a protein-DNA complex would allow researchers to characterize the specificity of newly solved structures, design proteins with desired binding properties, and explore the structural basis of recognition across diverse protein families.

This problem is particularly challenging because the relationship between three-dimensional structural geometry and sequence-level binding preference is complex. A reliable cross-family predictor must capture both direct base contacts (groove readout) and the geometric shape of the DNA helix (shape readout), integrating multiple types of spatial information simultaneously.

## 2. Task Objective

Given a three-dimensional protein-DNA complex structure, predict the position weight matrix (PWM) describing the protein's DNA-binding preference.

*   **Input**: For each test complex, a NumPy `.npz` file containing a structural representation of the protein-DNA complex:
    - Protein atom graph: node positions (`V_prot`, shape (N_prot, 3)), atom features (`X_prot`, shape (N_prot, 13)), edge indices (`E_prot`, shape (N_edges, 2)), orientation vectors (`prot_vectors`, shape (N_prot, 3))
    - DNA sym-helix representation: position grid (`V_dna`, shape (L, 11, 3)), base-step features (`X_dna`, shape (L, 14)), per-point features (`X_dna_point`, shape (L×11, 31)), orientation vectors (`dna_vectors`, shape (L, 11, 3)), strand validity mask (`dna_mask`, shape (2, L))
    - Co-crystal DNA sequence: one-hot encoded (`Y_hard`, shape (2, L, 4))
    - Auxiliary metadata: `contacts`, `prot_feature_names`, `dna_feature_names`
    - L is the number of base-pair positions in the structure (varies per complex).

*   **Output**: For each test complex, a predicted position weight matrix as a NumPy array of shape (L, 4) and dtype float32, where L = `V_dna.shape[0]` (the number of base-pair positions in the DNA helix). Each row must sum to 1.0 (probability distribution over A, C, G, T). The evaluation compares predictions at valid helix positions (identified by `dna_mask[0]` in the input) against the reference PWM.

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: MAE
*   **Definition**: Mean Absolute Error between the predicted and reference PWM, averaged over all positions and nucleotide columns. For a PWM with M positions and 4 nucleotide columns, MAE = (1/M) × Σ_i Σ_b |Y_ib − Ŷ_ib|. Lower is better.

### Other Metrics
*   **RMSE**: Root Mean Squared Error between predicted and reference PWM. RMSE = sqrt((1/M) × Σ_i Σ_b (Y_ib − Ŷ_ib)²). Lower is better.
