# ENV_03 — GEFCom2014-W Probabilistic Wind Power Forecasting

## 1. Scientific Problem

Use weather inputs for one wind-farm day to predict the 0.01–0.99 quantiles of normalized wind power for the 24 hours of that day, and evaluate probabilistic forecast quality.

## 2. Inputs and Targets

Each wind-farm day has 24 rows of weather records as input, with the corresponding 24 hours of normalized actual power output as targets. Submit 24×99 quantile predictions per sample. The original 24:00 record remains in the current sample; NaN values in the source targets do not cause the entire daily sample to be removed.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample consists of the 24 rows for one wind-farm day; one query provides all 24 hourly targets for that wind-farm day, without separate charges per hour or quantile.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 7,000 |
| Initially visible labels D0 = floor(0.1 N) | 700 |
| Unlabeled candidate pool Q | 6,300 |
| Maximum additional labels over the entire process floor(0.5 N) | 3,500 |
| Official V | 0 |
| Final T | 310 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

The training period is 2012-01-01 through 2013-11-30, and T is December 2013.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **mean_pinball_loss**, **lower is better**. For each quantile, compute pinball loss over hours with nonmissing ground truth, then weight the 99 quantiles equally; report the number of hours with missing targets separately.

## 6. Prediction Submission

`predictions/values.npy`, values.npy: (n_eval,24,99), with the last axis corresponding in order to 0.01..0.99; all predictions must be finite.

Submit an array of shape (n_eval,24,99), covering all 24×99 positions even when targets are missing, with all predictions finite. Quantiles should describe the same conditional distribution; the current rules do not treat quantile crossing as a separate rejection criterion.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; `n_eval` is the number of samples in this evaluation, not the training pool N. If the order is changed during inference, restore the global order before submission. Row i of values.npy corresponds to sample i.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="ENV_03", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"ENV_03","split":"T","count":310}
```
