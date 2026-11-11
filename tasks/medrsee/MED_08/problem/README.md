# MED_08 — BUS-BRA Breast Ultrasound Lesion Segmentation

## 1. Scientific Problem

Segment lesions in breast ultrasound images, restoring the final mask to the original dimensions of each image.

## 2. Inputs and Targets

The input is a grayscale breast ultrasound image at its original dimensions; output a mask of the same dimensions with 0 for background and 1 for lesion. Task inputs do not include Pathology, BIRADS, or BBOX from the source CSV.

See [data_description.md](data_description.md) for fields, channels, the objects read, and transformations already applied. Choose your own model, feature processing, and query strategy.

## 3. Samples and Label Budget

One sample is one ultrasound image, and label authorization is charged per image. group_id comes from Case in the source data; querying one image from a Case does not automatically reveal labels for other images from that Case.

| Item | Number of samples |
|---|---:|
| Fixed training pool N = D0 + Q | 1,499 |
| Initially visible labels D0 = floor(0.1 N) | 149 |
| Unlabeled candidate pool Q | 1,350 |
| Maximum additional labels over the entire process floor(0.5 N) | 749 |
| Official V | 0 |
| Final T | 376 |

Initially, only D0 training labels are available; Q labels require authorization for each sample. Q inputs can be read. Repeated queries for the same Q sample incur no additional charge, and querying one sample does not automatically authorize other samples in the same group. Fixed membership follows the existing splits.csv. The additional-label limit in the table is calculated from the training pool N, is shared across the entire task, and does not reset between generations.

## 4. Validation and Testing

fold1 is the fixed project T, and fold2–5 form the training pool; the same Case remains on one side of the split.

Select exact acquired sample IDs with `select_validation`. Each call replaces internal validation; pass `[]` to clear it. Choose or write your own splitting tools and revise the selection during research. The host does not expand groups or dates, impose a validation size, or diagnose the split. Newly queried labels remain training members until explicitly selected.

Use only authorized labels and record actual fitting membership and checkpoint ancestry. Any overlap with evaluated validation members must remain visible in the experiment record. Validation submissions cover all currently authorized validation members. T is evaluated only once, after the final model has been frozen, and is not used for model selection, hyperparameter tuning, or subsequent improvement.

## 5. Evaluation Metric

Primary metric: **patient_mean_dice**, **higher is better**. First compute lesion Dice per image, weight images equally within each Case (group_id), and then weight Cases equally. Retain the existing equal-patient/Case weighting; score 1 when both ground truth and prediction are empty, and 0 when only one is empty.

## 6. Prediction Submission

`predictions/000000.npz`, `000001.npz`, …; each file contains a scalar string `sample_id`; prediction_0: (H,W), at native dimensions, with classes 0/1.

Submit a 0/1 integer mask at the original H×W dimensions for each image; resolution may be changed during training, but must be restored to that image's original dimensions before submission.

Order all outputs by the original `ordinal` in ascending order within the set currently being evaluated; if the order is changed during inference, restore the global order before submission. Number per-sample NPZ files consecutively from `000000.npz`.

Write `predictions.json` and the `predictions/` data described above directly to `OUTPUT_DIR` in both phases; the host sets a separate `OUTPUT_DIR` for verified validation and for final inference, so do not append `validation/`. Validation metadata uses task="MED_08", split="validation", and count=the current number of validation members; final metadata is as follows:

```json
{"task":"MED_08","split":"T","count":376}
```
