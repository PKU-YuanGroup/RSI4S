# MED_03 — MSD Liver and Tumor Segmentation

## 1. Scientific Problem

Segment the liver and liver tumors from abdominal CT, producing a volume of three mutually exclusive class labels.

## 2. Inputs and Targets

The input is a complete abdominal CT. Output mutually exclusive labels: 0 for background, 1 for nontumor liver regions, and 2 for tumor. The liver evaluation region is 1∪2, and the tumor evaluation region is 2.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one complete CT case; one query provides the label volume for the entire case, without charges per slice or cropped patch.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 118 |
| Initially visible labels D0 = floor(0.1 N) | 11 |
| Unlabeled candidate pool Q | 107 |
| Maximum additional labels over the entire process floor(0.5 N) | 59 |
| Official V | 0 |
| Final T | 13 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Of 131 labeled cases, 118 form the training pool and 13 form the fixed project T; the official unlabeled test set is not used.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **structure_dice**, **higher is better**. Merge labels 1/2 for the liver region and use label 2 for tumor; weight the two structures equally, then weight cases equally. MONAI ignore_empty=False: 1 when both are empty, 0 when only one is empty.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0 is a native three-dimensional label array, accompanied by affine_0.

Submit one volume of mutually exclusive 0/1/2 labels per case, rather than two separate masks that can overlap. Cropping or resampling may be used during training; final prediction_0 must be restored to the original case shape and accompanied by the input-provided affine_0.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_03", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_03","split":"T","count":13}
```
