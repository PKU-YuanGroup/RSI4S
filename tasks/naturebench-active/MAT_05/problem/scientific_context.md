# Upconverting Nanoparticle Emission Intensity Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: sunset_1. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Lanthanide-doped upconverting nanoparticles (UCNPs) are optically active inorganic structures that absorb lower-energy photons (typically in the near-infrared) and re-emit higher-energy photons (visible or ultraviolet). Tuning their emission — particularly boosting the UV/blue portion of the spectrum — is central to applications in deep-tissue bioimaging, photodynamic therapy, anti-counterfeiting, 3D volumetric displays, and photocatalysis. Because each nanoparticle is built from a concentric core plus zero or more doped shell regions, with independent choices of region thickness, dopant element (e.g., Yb, Er, Nd), and dopant concentration, the design space is combinatorially large and strongly non-additive in its optical response: even a simulated evaluation of a single candidate design is expensive, so exhaustive search is impractical.

This task formulates the design-to-property mapping as a supervised regression: given only the structural and compositional parameters of a candidate UCNP, predict a scalar that summarizes the UV/blue (≈300–450 nm) emission intensity produced under a fixed excitation condition. A fast, accurate surrogate enables rapid in-silico screening and gradient-based optimization of UCNP heterostructures without re-running expensive stochastic physics simulations for every candidate.

## 2. Task Objective

Given the structural and compositional description of a UCNP (concentric spherical regions with radii, and per-region lanthanide dopant concentrations), predict a single scalar emission-intensity target for each test nanoparticle. The task has two test splits per instance: an in-distribution (IID) split drawn from the training distribution (0–3 shells) and an out-of-distribution (OOD) split composed of particles with a shell count not seen during training (4 shells).

*   **Input**: Per test nanoparticle, a structured record containing a list of concentric `SphericalConstraint` objects (each with a `radius` in angstroms), a list of `dopant_specifications` `[layer_index, concentration, element, host]`, a per-layer `dopant_concentration` dictionary list, a list of energy-level counts `n_levels`, and a record identifier `_id.oid`. See the dataset files for exact schema.
*   **Output**: For each test nanoparticle (identified by `_id.oid`), a single real-valued scalar `target` equal to `log10(I + 100)`, where `I` is the total simulated emission intensity summed over the 300–450 nm wavelength range (i.e., `I = sum of wavelength_spectrum_y at bin centers with 300 < |x| < 450` using the sign convention of the dataset).

## 4. Evaluation Metrics

### Primary Metric
*   **Name**: Mean Squared Error (MSE) on `log10(I + 100)`
*   **Definition**: `MSE = (1/N) * Σ (target_pred − target_ref)^2`, where `target_pred` is the predicted scalar and `target_ref = log10(I_ref + 100)` is the reference target derived from the simulated spectrum. Lower is better. The metric is reported separately for the IID and OOD test splits, and is multiplied by 10³ for readability.
