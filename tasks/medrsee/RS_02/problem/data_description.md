# RS_02 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One 240×240 crop; group_id is the source-scene identifier, and the same scene may contain multiple crops charged independently.

Native training pool 1,433, D0 143, Q 1,290, official V 642, T 728. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`input_0` is an `(11,240,240)` float32 array. The 11 channels are read in the established order of the source selector; the fixed release files contain the following two sets of center-wavelength labels (nm), so the first set must not be described as the sole wavelength sequence for all crops.

| Channel position | Release label A | Release label B |
|---:|---:|---:|
| 1 | 442 | 443 |
| 2 | 492 | 492 |
| 3 | 559 | 560 |
| 4 | 665 | 665 |
| 5 | 704 | 704 |
| 6 | 739 | 740 |
| 7 | 780 | 783 |
| 8 | 833 | 833 |
| 9 | 864 | 865 |
| 10 | 1610 | 1614 |
| 11 | 2186 | 2202 |

Inputs are the released `rhorc` band values; the reader does not divide them again by a fixed reflectance scale factor. The wavelength table provides supplementary spectral information; the Agent decides whether to use it explicitly, and no spectral interpolation or alignment to common center wavelengths is required. Current batches retain the 11-channel array in the established order, without separately exporting wavelength labels for each crop; models can use this channel order directly.

## 3. Labels and Transformations Already Applied

`label_0` is a `(240,240)` integer mask. Source 0→−1, source 1–15→internal 0–14. −1 marks unlabeled/ignored pixels, not background or Marine Water. One query provides a crop's existing sparse annotations, not complete dense ground truth.

| Internal label | Source label | Class |
|---:|---:|---|
| 0 | 1 | Marine Debris |
| 1 | 2 | Dense Sargassum |
| 2 | 3 | Sparse Floating Algae |
| 3 | 4 | Natural Organic Material |
| 4 | 5 | Ship |
| 5 | 6 | Oil Spill |
| 6 | 7 | Marine Water |
| 7 | 8 | Sediment-Laden Water |
| 8 | 9 | Foam |
| 9 | 10 | Turbid Water |
| 10 | 11 | Shallow Water |
| 11 | 12 | Waves & Wakes |
| 12 | 13 | Oil Platform |
| 13 | 14 | Jellyfish |
| 14 | 15 | Sea snot |

The 11 bands are aligned by nearest neighbor to the 240×240 reference grid of the 492 nm band; when a geotransform is missing, nearest-neighbor resizing is applied assuming the same spatial coverage. The current reader has no uniform NoData imputation step and does not guarantee that invalid pixels have been filled with zeros. Missing inputs and label −1 are different concepts.

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
| group_id | group_id identifies the scene; official V/T retains the source-scene split. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

Initial D0 labels and this task's existing official V labels are available, while Q requires authorization; V is used only for validation. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Class mappings use the authors' version fixed by this project. The two wavelength sets come from the fixed source selector.
