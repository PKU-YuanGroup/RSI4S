# ENV_02 — CAMELS-GB Daily Runoff Reconstruction

## 1. Scientific Problem

Use the meteorological sequence up to and including the target day and static catchment attributes to reconstruct runoff per unit area for that catchment on the target day. Target-day meteorological data may be used; weather after the target day must not be used.

## 2. Inputs and Targets

Use a meteorological sequence covering the 365 days before the target day plus the target day, 366 days in total, and static catchment attributes to predict target-day discharge_spec (mm/day). Target-day meteorological data may be used; runoff labels from neighboring days are not inputs. The target date can be obtained from the end of sample_id.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one target day for one catchment; one query authorizes only the runoff label for that day, not runoff for the entire 366-day window.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 7,347,592 |
| Initially visible labels D0 = floor(0.1 N) | 734,759 |
| Unlabeled candidate pool Q | 6,612,833 |
| Maximum additional labels over the entire process floor(0.5 N) | 3,673,796 |
| Official V | 0 |
| Final T | 2,409,999 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

The fixed training pool uses valid days from WY1972–2005, corresponding to 1971-10-01 through 2005-09-30; T uses valid days from WY2006–2015, corresponding to 2005-10-01 through 2015-09-30. Actual membership follows the fixed list; the same catchment may span the training and test periods.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **basin_macro_nse**, **higher is better**. Compute NSE = 1−SSE/SST for each catchment, then weight catchments equally; retain negative values and raise an error for catchments with constant ground truth.

## 6. Prediction Submission

`predictions/values.npy`，values.npy: (n_eval,1)，mm/day。

One target-day runoff scalar per row, in mm/day; all values must be finite.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; `n_eval` is the number of samples in this evaluation, not the training pool N. If the order is changed during inference, restore the global order before submission. Row i of values.npy corresponds to sample i.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="ENV_02", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"ENV_02","split":"T","count":2409999}
```
