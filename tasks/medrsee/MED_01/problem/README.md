# MED_01 — ACDC Cardiac MRI Segmentation

## 1. Scientific Problem

Perform multi-structure segmentation of cardiac MRI at end-diastole (ED) and end-systole (ES) for the same patient.

## 2. Inputs and Targets

The inputs are two three-dimensional MRI volumes from the same patient, at ED and ES. For each phase, predict 0 for background, 1 for the right ventricular cavity, 2 for myocardium, and 3 for the left ventricular cavity; one patient-level query provides labels for both phases.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample consists of the ED and ES phases of one patient; a query is counted once per patient, without separate charges per phase or slice.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 100 |
| Initially visible labels D0 = floor(0.1 N) | 10 |
| Unlabeled candidate pool Q | 90 |
| Maximum additional labels over the entire process floor(0.5 N) | 50 |
| Official V | 0 |
| Final T | 50 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Uses the official 100 training patients and 50 test patients.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **three_structure_dice**, **higher is better**. Official ACDC Dice, with structure order [3,1,2]; weight the three structures equally, weight ED/ES equally within each case, and finally weight cases equally.

Dice is computed on the corresponding prediction and ground-truth arrays, without using affine or spacing; the affine in the submission is used for the framework's consistency checks.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0 is ED and prediction_1 is ES, each with its native shape and affine_0/affine_1.

Submit two volumes of integer labels 0–3. Cropping or resampling may be used during training; for submission, restore ED and ES to their respective original grids and include the input-provided affine_0 and affine_1 for the corresponding phases.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_01", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_01","split":"T","count":50}
```
