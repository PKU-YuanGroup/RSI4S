# MED_07 — VerSe19 Vertebral Identity Segmentation

## 1. Scientific Problem

Segment vertebrae in spine CT and identify the fixed anatomical identity of each vertebra. Class identifiers must not be recoded according to the number or order of vertebrae present in the current scan.

## 2. Inputs and Targets

The input is spine CT; output background 0 and the fixed vertebral identity codes from the released data. Allowing integers 0–28 does not mean there are 28 actual supervised foreground classes; do not merge foreground classes or recode them by their order of appearance in the scan. See the data description for the full table of anatomical codes.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one complete scan; one query provides the label volume for that scan, without revealing labels for other scans of the same source subject.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 80 |
| Initially visible labels D0 = floor(0.1 N) | 8 |
| Unlabeled candidate pool Q | 72 |
| Maximum additional labels over the entire process floor(0.5 N) | 40 |
| Official V | 40 |
| Final T | 40 |

Initially, D0 training labels are available; official V labels are also available, but only for validation and without counting against the additional Q-label budget. Q labels still require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Uses the official train/validation/test split of 80/40/40 scans. Official V is used directly for validation, without creating another V. The same source subject in the released data does not span partitions.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

**MED_07 validation inference batch size.** Use `validation_batch_size=20` as the default and upper limit for MED_07 across Agent generations. This guidance applies only to MED_07 validation inference. Use a smaller batch when memory requirements demand it or for the final incomplete batch; do not increase it above 20 to use spare GPU memory. This is the number of examples passed to the model in one forward or prediction call, not the `fetch_data` page size or the number of validation members.

Process validation batches sequentially, without overlapping validation workers or keeping multiple batches on the GPU. Save each batch's predictions and release temporary tensors before processing the next batch. Evaluate every required validation member and preserve global `ordinal` order in the submission. The batch limit reduces resource contention but does not impose a GPU memory quota.

## 5. Evaluation Metric

Primary metric: **structure_dice**, **higher is better**. For each scan, take the union of nonzero identities present in the ground truth and prediction, and weight per-identity Dice equally; then average within each fixed group_id and weight groups equally. Retain the existing score aggregation, without recoding by order of appearance in the scan.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0 contains native three-dimensional integer labels, accompanied by affine_0.

Submit a native three-dimensional volume of integer identity labels 0–28 and the input-provided affine_0; restore the original grid and preserve the anatomical meaning of the codes.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_07", split="V", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_07","split":"T","count":40}
```
