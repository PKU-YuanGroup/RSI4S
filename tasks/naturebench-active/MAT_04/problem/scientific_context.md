# Polymer Property Prediction

The scientific problem, task objective, and metrics below are excerpts from the native
NatureBench README. Included instances: ea, eat, eea, egb, egc, ei, eps, ip, mar1, mar2, nc, opv, pe. Native instances outside this list are
excluded. Current data access and labeling rules are in README.md and data_description.md.
Validation uses the task's native metric definitions; Test feedback is terminal-only.

## 1. Scientific Problem

Polymers are long-chain molecules built by repetition of one or more monomer units. Their macroscopic physical, electronic and electrochemical properties — band gap, ionization/electron energies, refractive index, dielectric constant, atomization energy, photovoltaic efficiency, ionic conductivity, and so on — determine whether a candidate material is suitable for applications such as organic solar cells, solid-state batteries, optical coatings, and insulating films. Measuring these properties experimentally or computing them from first-principles quantum chemistry is expensive, so the ability to predict polymer properties directly from the repeating-unit chemical structure is a central problem in computational materials discovery.

A key challenge is that polymer structures are inherently periodic and often involve copolymers with mixed monomers, composition ratios and chain topology (alternating, block, random). A single repeating unit SMILES, optionally combined with composition and topology descriptors or external physical conditions, must be mapped to a real-valued scalar property. This task benchmarks a regression model's ability to make such mappings across a diverse set of polymer property targets.

## 2. Task Objective

For each instance, given a set of labeled polymer samples (with features and target values) plus unlabeled test samples (features only), train a regression model and predict the target scalar for every test sample.

* **Input**: CSV files in `problem/data/{instance}/` containing labeled training (`train.csv`) and validation (`val.csv`) data, an unlabeled test set (`test.csv`), and an auxiliary large-scale unlabeled polymer corpus (`PI1M.csv`). Features include SMILES strings (with `*` denoting repeating-unit connection points) and, depending on the instance, additional composition ratios, chain-type labels, or physical/formulation descriptors.
* **Output**: For each instance, a 1-D array of predicted scalar targets — one prediction per row of `test.csv`, in the same order.

## 4. Evaluation Metrics

### Primary Metric
* **Name**: RMSE (root-mean-squared error)
* **Definition**: `RMSE = sqrt(mean((y_pred - y_true)^2))`. Computed per instance in the native units of each target property. Lower is better.
