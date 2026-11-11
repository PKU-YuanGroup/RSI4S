# RS_02 — MADOS Marine Floating Material and Pollution Identification

## 1. Scientific Problem

Use Sentinel-2 multispectral imagery for sparse semantic segmentation of marine floating material, pollution, and related marine classes.

## 2. Inputs and Targets

Use multispectral crops with 11 channels to predict 15 marine classes. Source labels 1–15 map to internal labels 0–14; source 0 maps to −1, meaning unlabeled/ignored, not background or Marine Water. One query provides the existing sparse annotations, not complete dense ground truth.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one 240×240 crop. One query provides only the existing sparse annotations for that crop, not complete dense ground truth, and does not reveal labels for other crops from the same scene.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 1,433 |
| Initially visible labels D0 = floor(0.1 N) | 143 |
| Unlabeled candidate pool Q | 1,290 |
| Maximum additional labels over the entire process floor(0.5 N) | 716 |
| Official V | 642 |
| Final T | 728 |

Initially, D0 training labels are available; official V labels are also available, but only for validation and without counting against the additional Q-label budget. Q labels still require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Uses the official scene split; official V contains 642 crops and is used directly for validation, without creating another V.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **macro_f1**, **higher is better**. Pool all valid pixels and call the authors' Evaluation, computing macro F1 over the union of ground-truth and predicted classes.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0 for each sample: (240,240), with classes 0..14.

Submit an integer class in 0–14 for every pixel; −1 is only for ignored ground-truth regions and is not a valid submission class. Outputs retain the original 240×240 grid; do not submit probabilities or logits.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="RS_02", split="V", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"RS_02","split":"T","count":728}
```
