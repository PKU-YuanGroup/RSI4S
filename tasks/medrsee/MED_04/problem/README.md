# MED_04 — LUNA16 Pulmonary Nodule Detection

## 1. Scientific Problem

Detect pulmonary nodules in complete chest CT scans, and submit candidate nodule centers as XYZ coordinates in millimeters in the original physical space, together with confidence scores.

## 2. Inputs and Targets

The input is a complete chest CT. included annotations define target nodules, while excluded annotations are handled according to the established rules; one query provides both annotation sets for the entire scan. Submit original physical XYZ coordinates in millimeters and confidence scores; annotation diameters are not prediction submission columns.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one complete CT scan; one query provides both the included and excluded annotation sets for that scan.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 800 |
| Initially visible labels D0 = floor(0.1 N) | 80 |
| Unlabeled candidate pool Q | 720 |
| Maximum additional labels over the entire process floor(0.5 N) | 400 |
| Official V | 0 |
| Final T | 88 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

subset0–8 form the training pool, and subset9 is the fixed project T.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **CPM**, **higher is better**. Uses official evaluateCAD nodule-radius matching and excluded-annotation handling; average sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, and 8 FP/scan, without bootstrap.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0: (K,4), K≤100; columns are physical X/Y/Z (mm) and confidence in [0,1]; submit (0,4) when there are no candidates.

Each candidate consists of exactly four numeric values [X_mm,Y_mm,Z_mm,confidence], with confidence in [0,1] and all values finite; at most 100 candidates per scan, with (0,4) submitted when there are none. Coordinates from cropping or resampling during training must be mapped back to the original physical coordinates; voxel indices must not be submitted.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_04", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_04","split":"T","count":88}
```
