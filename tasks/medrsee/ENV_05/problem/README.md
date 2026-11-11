# ENV_05 — Battery Capacity Estimation from Relaxation Curves

## 1. Scientific Problem

Estimate the current discharge capacity of the same continuous diagnostic cycle segment from the voltage relaxation curve after high-voltage charging.

## 2. Inputs and Targets

Use variable-length relaxation curves after high-voltage charging and operating-condition attributes to predict the current discharge capacity (Ah) of the same continuous diagnostic interval. The target maximum capacity occurs after the selected relaxation segment. Inputs do not include the discharge-capacity column, and no common sampling rate, number of points, or duration is assumed across chemistries or samples.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one continuous diagnostic cycle segment of one cell and its selected relaxation curve; one query provides the capacity of this sample, without revealing capacities for other cycles of the cell.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 53,823 |
| Initially visible labels D0 = floor(0.1 N) | 5,382 |
| Unlabeled candidate pool Q | 48,441 |
| Maximum additional labels over the entire process floor(0.5 N) | 26,911 |
| Official V | 0 |
| Final T | 5,110 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

Cells are held out for testing within each chemistry.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **group_equal_mae**, **lower is better**. Compute MAE within each cell, then weight cells equally; also report RMSE for each chemistry.

## 6. Prediction Submission

`predictions/values.npy`，values.npy: (n_eval,1)，Ah。

One current-capacity scalar per row, in Ah; all values must be finite. Do not apply the mAh→Ah conversion again.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; `n_eval` is the number of samples in this evaluation, not the training pool N. If the order is changed during inference, restore the global order before submission. Row i of values.npy corresponds to sample i.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="ENV_05", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"ENV_05","split":"T","count":5110}
```
