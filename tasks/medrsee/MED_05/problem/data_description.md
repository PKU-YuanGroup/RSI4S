# MED_05 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One 256×256 histopathology patch. group_id and holdout_group are source-image identifiers, not verified patient identities.

Native training pool 4,537, D0 453, Q 4,084, official V 0, T 444. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is a `(3,256,256)` float32 RGB array with channels in R/G/B order; it has already been divided by 255 during loading.

## 3. Labels and Transformations Already Applied

`label_0` is a `(256,256,2)` integer array: index 0 of the last axis holds instance IDs, and index 1 holds type IDs.

| Type ID | Meaning |
|---:|---|
| 0 | Background; instance ID is also 0 |
| 1 | Neutrophils |
| 2 | Epithelial cells |
| 3 | Lymphocytes |
| 4 | Plasma cells |
| 5 | Eosinophils |
| 6 | Connective tissue cells |

For example, two adjacent epithelial cells may use instance IDs 8 and 23, respectively, while both have type 2; they must not be merged into one instance because their types match. Instance IDs need not be consecutive; each instance can have only one type.

Images are converted from H×W×RGB to RGB×H×W, cast to float32, and divided by 255; labels retain H×W×2 layout and are converted to integers. The reader does not merge instances, normalize staining, or augment data.

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
- `inputs[i]` is a dictionary of arrays for this sample, containing input_0; after authorization it also contains label_0.
- The third return value, `labels`, is `None`, not an array holding labels collectively.

Each image sample is saved as one NPZ file, located through file in `samples.json`; there is no need to guess paths.

Unauthorized samples have no label keys; do not interpret missing label keys as "all background."

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | Patches from the same source image remain on the same side of the training/validation split. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Codes follow the original CoNIC instance/type annotations.
