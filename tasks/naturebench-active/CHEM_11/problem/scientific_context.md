# Electronic Circular Dichroism Spectrum Peak Property Prediction

## 1. Scientific Problem

Electronic circular dichroism (ECD) spectroscopy is a fundamental analytical technique for determining the absolute configuration of chiral molecules. Chiral molecules -- molecules that are non-superimposable on their mirror images -- play critical roles in asymmetric catalysis, drug discovery, and functional materials science. The ECD spectrum encodes the differential absorption of left- and right-circularly polarized light, producing characteristic peaks whose properties (number, positions, and signs) reveal the stereochemical features of a molecule.

Traditional computation of ECD spectra requires multi-stage quantum chemical calculations including conformational search, structural optimization, and time-dependent density functional theory (TD-DFT), which are computationally expensive (hours per molecule). Developing machine learning models that can accurately predict ECD peak properties directly from molecular structure would dramatically accelerate chirality determination workflows.

## 2. Task Objective

Given a chiral molecule represented by its structural features (atom-bond-angle graph and molecular descriptors), predict the peak properties of its ECD spectrum.

*   **Input**: Molecular structure data including atom-level features (atomic number, chirality tag, degree, valence, charge, hybridization, aromaticity, hydrogen count), bond-level features (direction, type, ring membership, length), bond-angle features, and precomputed molecular descriptors (1,826 features). Molecules are provided as graph structures with atom-bond and bond-angle connectivity. Test molecules are specified in `x_test.json` with references to the full molecular data files.
*   **Output**: For each test molecule, predict three peak properties:
    1. **Peak number**: The number of peaks in the ECD spectrum (integer)
    2. **Peak positions**: The position of each peak as an index into a 20-point sampled spectrum (list of integers, 0-19)
    3. **Peak symbols**: The sign of each peak's Cotton effect (1 = positive, 0 = negative) (list of integers)

## 4. Evaluation Metrics

### Primary Metric
*   **Symbol-Acc**: The matching accuracy of peak symbols (Cotton effect signs) between predicted and ground-truth ECD spectra. For each sample, peak symbols are compared position-by-position up to the minimum of the predicted and ground-truth peak counts. The metric is the ratio of correctly matched symbols to the total number of ground-truth peaks, averaged across all test samples. Higher is better.

### Other Metrics
*   **Number-RMSE**: Root mean square error of the predicted peak count versus the ground-truth peak count, computed per-sample and aggregated. Lower is better.
*   **Position-RMSE**: Root mean square error of predicted peak positions versus ground-truth peak positions. Peaks are aligned by order; when peak counts differ, missing peaks are padded with the sequence length (20). RMSE is computed over all peaks across all test samples. Lower is better.
