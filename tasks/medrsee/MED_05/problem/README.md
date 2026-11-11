# MED_05 — CoNIC Nucleus Instance Segmentation and Classification

## 1. Scientific Problem

Perform nucleus instance segmentation on H&E histopathology patches and assign each nucleus to one of six cell types.

## 2. Inputs and Targets

The input is a 256×256 RGB histopathology patch; output an instance map and a cell-type map with six classes. Adjacent nuclei of the same type must still have different instance IDs; IDs need not be consecutive, and each instance may have only one type. See the data description for class codes.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one histopathology patch; one query provides the instance map and type map for that patch, without charges per nucleus.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 4,537 |
| Initially visible labels D0 = floor(0.1 N) | 453 |
| Unlabeled candidate pool Q | 4,084 |
| Maximum additional labels over the entire process floor(0.5 N) | 2,268 |
| Official V | 0 |
| Final T | 444 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

The outer split separates source images by group_id; patches from the same source image remain on the same side.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **mPQ+**, **higher is better**. For each of the six classes, accumulate TP/FP/FN/IoU over all patches, then compute per-class PQ and the average over the six classes; retain the authors' 1e-6 term.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0: (256,256,2), containing nonnegative integers; instance IDs need not be consecutive.

Submit a (256,256,2) array of nonnegative integers. Background positions in the instance and type maps must match; every nonzero instance must have exactly one type in 1–6. Do not merge nuclei of the same type into a single semantic region.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_05", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_05","split":"T","count":444}
```
