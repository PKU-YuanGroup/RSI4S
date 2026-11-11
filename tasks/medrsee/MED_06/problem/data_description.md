# MED_06 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One complete ceT1 case from the CrossMoDA ceT1 supervised subset. group_id and holdout_group are case identifiers.

Native training pool 95, D0 9, Q 86, official V 0, T 10. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is a native three-dimensional ceT1 MRI array, and `affine_0` is the 4×4 source affine; NIfTI array axes are retained, without an additional channel axis. Current inputs do not include hrT2.

## 3. Labels and Transformations Already Applied

`label_0` has the same shape as the input: 0 for background, 1 for vestibular schwannoma, and 2 for cochlea; one query provides labels for both structures.

NIfTI data and the affine are decoded, and labels are converted to integers; no automatic MRI intensity standardization or resampling is performed. After applying your chosen training preprocessing, predictions must still be restored to the original case grid.

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
- `inputs[i]` is a dictionary of arrays for this sample, containing input_0 and affine_0; after authorization it also contains label_0.
- The third return value, `labels`, is `None`, not an array holding labels collectively.

Each image sample is saved as one NPZ file, located through file in `samples.json`; there is no need to guess paths.

Unauthorized samples have no label keys; do not interpret missing label keys as "all background."

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | Of 105 source-domain ceT1 cases, 95 are for training and 10 form the project T; entire cases are kept separate. This is a supervised segmentation subset, not the full cross-modality challenge. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

The task scope is consistently defined as the CrossMoDA ceT1 supervised subset.
