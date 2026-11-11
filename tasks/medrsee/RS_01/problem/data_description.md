# RS_01 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One 128×128 patch; sample_id is the fixed patch identifier, and group_id uses the same patch identifier, which does not represent an independent landslide event.

Native training pool 3,799, D0 379, Q 3,420, official V 245, T 800. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

The input `input_0` is float32 with shape `(14,128,128)`, with axes for channels, height, and width.

| Channel position (starting from 1) | Content |
|---|---|
| 1–12 | B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, in that order |
| 13 | Slope, the slope variable |
| 14 | DEM, the elevation variable |

Spectral and terrain channels are provided on the numeric scales of the released files. Channel names do not mean the current values are already raw reflectance, angles, or meters; the precise scaling formulas applied before release have not yet been confirmed.

## 3. Labels and Transformations Already Applied

`label_0` is a `(128,128)` integer mask: 0 for non-landslide and 1 for landslide. One patch-level query provides labels for that patch.

HDF5 images are converted from H×W×C to C×H×W and cast to float32. The current reader neither fits nor applies normalization and does not additionally impute missing values. Determine your own preprocessing on the permitted training side and freeze it for validation and inference.

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
| group_id | This task uses the official patch split; group_id provides no guarantee of event-level separation. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

Initial D0 labels and this task's existing official V labels are available, while Q requires authorization; V is used only for validation. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

The channel order is listed in the table above.
