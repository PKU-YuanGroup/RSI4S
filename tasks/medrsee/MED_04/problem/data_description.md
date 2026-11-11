# MED_04 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One complete chest CT scan. sample_id is seriesuid; group_id and holdout_group use the scan identifier.

Native training pool 800, D0 80, Q 720, official V 0, T 88. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is a three-dimensional CT decoded from MHD+RAW, with array axes in X, Y, Z order. `affine_0` is a 4×4 matrix mapping between voxel indices and original physical coordinates. Array indices and millimeter coordinates are not the same kind of value.

| Annotation field | Meaning | Type / unit |
|---|---|---|
| seriesuid | Scan identifier | String |
| coordX, coordY, coordZ | Center in the original physical space | Numeric, mm |
| diameter_mm | Annotated nodule diameter | Numeric, mm |

Numeric fields may still be strings after reading the original CSV and should be converted to numbers. Do not treat them as categorical text.

## 3. Labels and Transformations Already Applied

For authorized scans, `samples[i]["annotations"]` contains two lists, `[included, excluded]`, each entry being a dictionary of the fields in the table above. included contains target nodules, while excluded is ignored according to the scoring rules; no voxel segmentation mask is returned. One query authorizes both annotation sets for the entire scan. Diameter is an annotation field; prediction submissions still contain four columns: XYZ and confidence.

RAW is decoded using the original data type and converted to X/Y/Z axis order, without automatic CT windowing or resampling. Candidates obtained after cropping or resampling must be mapped back to the original physical XYZ coordinates in millimeters. Use the spatial reference of the current fixed data according to the provided source information.

## 4. How to Read a Batch

Obtain a batch path through `fetch_data`, then call `PROBLEM_DIR/reader.py`; the organizer's original mixed-content files are not directly accessible.

```python
import os, sys
sys.path.insert(0, os.environ["PROBLEM_DIR"])
from reader import load_batch
samples, inputs, labels = load_batch(batch_path)  # Path returned by fetch_data
i = 0
ct = inputs[i]["input_0"]
if samples[i]["labeled"]:
    included, excluded = samples[i]["annotations"]
```

`load_batch` returns `(samples, inputs, labels)`:

- `samples[i]` stores the sample ID, group, ordinal, and labeled authorization status.
- `inputs[i]` is a dictionary of arrays for this sample, containing input_0 and affine_0.
- The third return value, `labels`, is `None`, not an array holding labels collectively.

Each image sample is saved as one NPZ file, located through file in `samples.json`; there is no need to guess paths.

Authorized annotations are in `samples[i]["annotations"]`, not in `inputs[i]`. Unauthorized scans have no annotations key.

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | Entire scans are kept separate; subset0–8 are for training, and subset9 is the project T. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Corresponds to the current fixed LUNA16 scan set; the original annotation format and prediction format are described separately and must not be confused.
