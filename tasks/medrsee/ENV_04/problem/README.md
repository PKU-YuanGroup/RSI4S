# ENV_04 — ECOPROPHET GPP Productivity Estimation

## 1. Scientific Problem

Estimate daily gross primary productivity (GPP) at a site from daily meteorological data, soil moisture, and vegetation attributes, and evaluate generalization to unseen spatial groups of sites.

## 2. Inputs and Targets

Predict daily site-level GPP from meteorological data, soil moisture, and site attributes. The target Obs is in gC m⁻² day⁻¹ and comes from daytime flux partitioning; task data loading does not convert its units again. See the data description for detailed fields, product sources, and missing-data information.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one valid day at one site; one query provides one GPP label for that site-day, without revealing labels for other dates at the same site.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 140,002 |
| Initially visible labels D0 = floor(0.1 N) | 14,000 |
| Unlabeled candidate pool Q | 126,002 |
| Maximum additional labels over the entire process floor(0.5 N) | 70,001 |
| Official V | 0 |
| Final T | 16,025 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

The outer split separates connected groups of sites formed using 10 km proximity. The public holdout_group records this grouping; group_id identifies the site used for score aggregation. Internal validation membership is chosen by the Agent.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **group_equal_mae**, **lower is better**. Compute MAE within each site, then weight sites equally; this is not the overall MAE across all site-days.

## 6. Prediction Submission

`predictions/values.npy`，values.npy: (n_eval,1)，gC m^-2 day^-1。

One GPP scalar per row, in gC m⁻² day⁻¹; all values must be finite.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; `n_eval` is the number of samples in this evaluation, not the training pool N. If the order is changed during inference, restore the global order before submission. Row i of values.npy corresponds to sample i.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="ENV_04", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"ENV_04","split":"T","count":16025}
```
