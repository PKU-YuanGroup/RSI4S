# RS_01 — Landslide4Sense Landslide Segmentation

## 1. Scientific Problem

Use Sentinel-2 spectral and terrain information to identify landslide regions in 128×128 patches, producing a binary mask for each pixel.

## 2. Inputs and Targets

Use 14 channels of spectral and terrain inputs to predict a 128×128 landslide mask with labels 0/1. The reader does not normalize automatically; determine preprocessing on the permitted training side and freeze it for validation and inference.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one 128×128 patch; one query provides the landslide mask for that patch.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 3,799 |
| Initially visible labels D0 = floor(0.1 N) | 379 |
| Unlabeled candidate pool Q | 3,420 |
| Maximum additional labels over the entire process floor(0.5 N) | 1,899 |
| Official V | 245 |
| Final T | 800 |

Initially, D0 training labels are available; official V labels are also available, but only for validation and without counting against the additional Q-label budget. Q labels still require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Uses the official patch-level train/validation/test split; official V contains 245 patches and is used directly for validation, without creating another V.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **landslide_f1**, **higher is better**. Accumulate TP/FP/FN over all test pixels image by image, then compute F1 for the landslide class; do not average per-image F1.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0: (128,128), with classes 0/1.

Submit integer labels 0/1, not probabilities or logits. Resolution may be changed during training, but outputs must be restored to the original 128×128 grid.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="RS_01", split="V", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"RS_01","split":"T","count":800}
```
