# MED_01 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One patient, including two three-dimensional phases: ED (end-diastole) and ES (end-systole). Both group_id and holdout_group identify the patient.

Native training pool 100, D0 10, Q 90, official V 0, T 50. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

| Returned key | Content | Shape / meaning |
|---|---|---|
| input_0 | ED image | Native three-dimensional array |
| input_1 | ES image | Native three-dimensional array |
| affine_0 | Source affine for ED | 4×4 |
| affine_1 | Source affine for ES | 4×4 |

The arrays retain their respective NIfTI axis orders, without an additional channel axis; dimensions are those actually returned for each sample. The source affine is used for correspondence to the original grid and submission consistency.

**Additional spatial information:** Voxel spacing describes the physical distance between adjacent voxels along the three array axes and comes from pixdim/zooms in each phase's NIfTI header; ED and ES should each correspond to their own image. It can be used to choose resampling ratios or understand slice spacing. The Agent decides whether to use it; no fixed resampling method is required. Current batches provide arrays and affines, with no separate spacing key; source-header spacing may differ from the norms of the affine columns. Final submissions must still use the original array grid and retain the input affine.

## 3. Labels and Transformations Already Applied

One query provides both `label_0` (ED) and `label_1` (ES), each with the same shape as its corresponding input. Labels are 0 for background, 1 for the right ventricular cavity, 2 for myocardium, and 3 for the left ventricular cavity.

NIfTI data and source affines are decoded, and labels are converted to integers; no automatic resampling, MRI intensity standardization, or normalization fitting is performed. The Agent decides on cropping/resampling during training; restore ED and ES to their respective original grids for the final submission.

## 4. How to Read a Batch

Obtain a batch path through `fetch_data`, then call `PROBLEM_DIR/reader.py`; the organizer's original mixed-content files are not directly accessible.

```python
import os, sys
sys.path.insert(0, os.environ["PROBLEM_DIR"])
from reader import load_batch
samples, inputs, labels = load_batch(batch_path)  # Path returned by fetch_data
i = 0
x = inputs[i]["input_0"]
if samples[i]["labeled"]:
    y = inputs[i]["label_0"]
```

`load_batch` returns `(samples, inputs, labels)`:

- `samples[i]` stores the sample ID, group, ordinal, and labeled authorization status.
- `inputs[i]` is a dictionary of arrays for this sample, containing input_0, input_1, affine_0, and affine_1; after authorization it also contains label_0 and label_1.
- The third return value, `labels`, is `None`, not an array holding labels collectively.

Each image sample is saved as one NPZ file, located through file in `samples.json`; there is no need to guess paths.

Unauthorized samples have no label keys; do not interpret missing label keys as "all background."

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | Both phases and all slices of a patient share one group and must not span training and validation. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Uses source ED/ES images and labels for the current ACDC patients; spacing information is supplementary and adds no prediction submission fields or new scoring requirements.
