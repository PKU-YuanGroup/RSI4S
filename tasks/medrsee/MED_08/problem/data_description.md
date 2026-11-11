# MED_08 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One breast ultrasound image. group_id comes from the source Case, for example Case 22 corresponds to case22; holdout_group also uses Case.

Native training pool 1,499, D0 149, Q 1,350, official V 0, T 376. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is a `(1,H,W)` float32 grayscale image; during loading, it is converted to grayscale, given a channel axis, and divided by 255. H and W are the original dimensions of this image, without resizing to a common size. Inputs do not include Pathology, BIRADS, or BBOX from the source CSV.

## 3. Labels and Transformations Already Applied

`label_0` is an `(H,W)` integer mask: nonzero pixels in the source mask → lesion 1, all others → background 0. Each image is authorized independently; other images from the same Case do not automatically receive labels.

Grayscale conversion and division by 255 are performed during loading, and labels undergo a deterministic mapping from nonzero values to 1; no fixed image size is imposed, and no normalization is fitted. If images are resized during training, outputs must be restored to each image's original H×W dimensions.

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
| group_id | Separate by source Case: fold1 is the project T, and fold2–5 form the training pool. One Case may contain multiple images; the grouping unit and query unit differ. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Case mappings and fold assignments use the fixed BUS-BRA membership table.
