# MED_02 — BraTS2021 Four-Modality Brain Tumor Segmentation

## 1. Scientific Problem

Segment tumor subregions from four MRI modalities of adult glioma cases, comparing methods and annotation strategies on the same fixed data source. This instance uses the custom split `custom1000_251_v1` of 1251 publicly labeled cases, not the official challenge test set.

## 2. Inputs and Targets

Input channels are T1, contrast-enhanced T1 (T1ce), T2, and FLAIR, in that order. Output one mutually exclusive label volume: 0 background, 1 necrotic tumor core, 2 edema, and 4 enhancing tumor. Label 4 is not remapped to 3. See [data_description.md](data_description.md) for returned objects, array shapes, and transformations already applied. Choose your own model, preprocessing, and query strategy.

## 3. Samples and Label Budget

One sample is one complete case. One query provides its complete three-dimensional mask; modalities, slices, and training crops are not counted separately.

| Item | Number of cases |
|---|---:|
| Fixed training pool N=D0+Q | 1,000 |
| Initially visible labels D0=floor(0.1N) | 100 |
| Unlabeled candidate pool Q | 900 |
| Maximum additional labels floor(0.5N) | 500 |
| Official V | 0 |
| Fixed project T | 251 |

D0 labels are initially available. Q inputs can be read, but labels require authorization per case. Repeated queries for the same case do not consume additional budget, and the budget does not reset between generations. At most 600 cases can be labeled, including cases used for internal validation. Fixed membership follows the existing split CSV.

## 4. Validation and Testing

The organizer has fixed 1000 training cases and 251 T cases from the public 1251 cases, stratifying by three WT-volume bins and the presence of ET; D0 follows the same stratification. All modalities, masks, slices, and crops remain grouped by case.

There is no additional free labeled V. Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected. Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members.

All methods share the same fixed T and initial D0; do not change the split based on model scores. Each predeclared method is evaluated on T once after its final model and preprocessing are frozen. T is not used for model selection, tuning, or subsequent improvement.

## 5. Evaluation Metric

The primary metric is **region_dice**, higher is better. Compute Dice for three regions in each case:

| Region | Corresponding labels |
|---|---|
| WT (whole tumor) | Union of 1, 2, and 4 |
| TC (tumor core) | Union of 1 and 4 |
| ET (enhancing tumor) | 4 |

Average the three regions equally, then average cases equally. Score 1 when prediction and ground truth are both empty, and 0 when only one is empty. No automatic small-lesion removal is applied. Dice is computed on corresponding arrays; the affine is used for submission geometry checks.

## 6. Prediction Submission

Save one file per case as `predictions/000000.npz`, `000001.npz`, and so on, ordered by ascending fixed `ordinal` within the evaluated set. Each file contains a scalar string `sample_id`, native three-dimensional integer labels in `prediction_0`, and the input-provided 4×4 `affine_0`. Prediction labels must be 0/1/2/4. Restore any cropped or resampled outputs to the original case grid before submission.

Write `predictions.json` and `predictions/` directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_02", split="validation", and count=the current number of authorized validation cases. Final metadata is:

```json
{"task":"MED_02","split":"T","count":251}
```

Follow the framework's shared submission instructions for model freezing and the final inference entrypoint.
