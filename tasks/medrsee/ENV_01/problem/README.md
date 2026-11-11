# ENV_01 — GLORIA Chlorophyll Concentration Retrieval

## 1. Scientific Problem

Estimate the chlorophyll a concentration of an in situ observation from water remote-sensing reflectance spectra. The provided labels, submitted predictions, and evaluation all use the log10 concentration scale.

## 2. Inputs and Targets

Use spectra from 350–900 nm to predict the chlorophyll a concentration of an observation. An observation may have multiple replicate spectra; choose how to combine them and output a single scalar. Both the provided labels and submitted values are log10(Chla), with the concentration expressed in mg m⁻³; do not take the log again or apply an inverse transformation when submitting.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample corresponds to one in situ observation; one query provides one concentration label for that observation. Charges are not counted by the number of replicate spectra.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 4,412 |
| Initially visible labels D0 = floor(0.1 N) | 441 |
| Unlabeled candidate pool Q | 3,971 |
| Maximum additional labels over the entire process floor(0.5 N) | 2,206 |
| Official V | 0 |
| Final T | 718 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

The outer split holds out water-body name groups formed from Country_code and normalized Site_name.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **log10_chla_rmse**, **lower is better**. RMSE in log10 space over all observations, without averaging by water body.

## 6. Prediction Submission

`predictions/values.npy`, values.npy: (n_eval,1), on the log10(Chla) scale.

One log10(Chla) scalar per row; the concentration here is expressed in mg m⁻³. All submitted values must be finite.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; `n_eval` is the number of samples in this evaluation, not the training pool N. If the order is changed during inference, restore the global order before submission. Row i of values.npy corresponds to sample i.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="ENV_01", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"ENV_01","split":"T","count":718}
```
