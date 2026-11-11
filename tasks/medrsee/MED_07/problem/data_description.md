# MED_07 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One spine CT scan; the same source subject may have multiple scans. Queries are charged per scan, and group_id and holdout_group correspond to the source subject in the released data.

Native training pool 80, D0 8, Q 72, official V 40, T 40. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is a native three-dimensional CT array, and `affine_0` is the 4×4 source affine; the original array axes and scan grid are retained, without an additional channel axis.

## 3. Labels and Transformations Already Applied

`label_0` has the same shape as the input and uses fixed anatomical identity codes:

| Label | Anatomical structure |
|---:|---|
| 0 | Background |
| 1–7 | C1–C7 |
| 8–19 | T1–T12 |
| 20–25 | L1–L6 |
| 26 | Sacrum; not annotated in this dataset |
| 27 | Coccyx; not annotated in this dataset |
| 28 | T13 |

Allowing integers 0–28 does not mean that all 28 foreground classes are annotated in this dataset. Codes must not be reordered by their order of appearance within a scan; source subject identifiers do not additionally establish hospital patient identities.

NIfTI data and the affine are decoded, and labels are converted to integers; vertebral identities are not changed, foreground classes are not merged, and no automatic resampling or normalization is performed.

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
| group_id | Official train/V/T contains 80/40/40 scans, corresponding to 67/37/37 source subjects, with no subject spanning partitions. Querying one scan does not automatically authorize other scans of the same subject. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

Initial D0 labels and this task's existing official V labels are available, while Q requires authorization; V is used only for validation. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

The codes follow the authors' version recorded by this project.
