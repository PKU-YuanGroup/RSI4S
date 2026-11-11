# MED_06 — CrossMoDA ceT1 supervised subset

## 1. Scientific Problem

Within the CrossMoDA ceT1 supervised subset, use contrast-enhanced T1 MRI for supervised segmentation of vestibular schwannoma and cochlea. This is a ceT1 supervised subtask, not the full original CrossMoDA cross-modality domain adaptation challenge.

## 2. Inputs and Targets

The input is native three-dimensional ceT1 MRI; predict 0 for background, 1 for vestibular schwannoma, and 2 for cochlea. This supervised subtask does not require hrT2 or unsupervised domain adaptation.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one complete ceT1 case; one query provides both tumor and cochlea labels for that case.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 95 |
| Initially visible labels D0 = floor(0.1 N) | 9 |
| Unlabeled candidate pool Q | 86 |
| Maximum additional labels over the entire process floor(0.5 N) | 47 |
| Official V | 0 |
| Final T | 10 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Of 105 source-domain ceT1 cases, 95 form the training pool and 10 form the fixed project T.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **structure_dice**, **higher is better**. Weight tumor/cochlea Dice equally, then weight cases equally; MONAI gives 1 when both are empty and 0 when only one is empty.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0 contains native three-dimensional labels, accompanied by affine_0.

Submit a native three-dimensional volume of integer labels 0/1/2 and the input-provided affine_0; if cropping or resampling is used during training, restore the original case grid before submission.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_06", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_06","split":"T","count":10}
```
