# ENV_01 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One in situ observation; it may correspond to multiple replicate spectra but has only one target value. group_id identifies the group formed from the country code and normalized water-body name.

Native training pool 4,412, D0 441, Q 3,971, official V 0, T 718. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`inputs[i][0]` is a dictionary of named columns for observation i. Each column has length m, the number of spectra for that observation.

| Field | Meaning | Unit / structure |
|---|---|---|
| Rrs_350 … Rrs_900 | Remote-sensing reflectance from 350 to 900 nm in 1 nm steps | 551 columns; Rrs in sr⁻¹; each column has length m |

Choose how to combine multiple spectra from the same observation; no averaging method is fixed in advance. Missing wavelength values remain NaN.

## 3. Labels and Transformations Already Applied

The original `Chla` is a concentration expressed in mg m⁻³. The task export stage computes `log10(Chla)`; `labels[i,0]` returns this log value, and predictions use the same scale. Reading the native table and exporting task batches are two separate steps; do not take the log again or convert back to concentration when submitting.

The spectral rows corresponding to the fixed observations are selected, retaining all 551 columns and replicate rows. No spectral averaging, interpolation, normalization, or missing-value imputation is applied. Only authorized targets undergo the log10 transformation.

## 4. How to Read a Batch

Obtain a batch path through `fetch_data`, then call `PROBLEM_DIR/reader.py`; the organizer's original mixed-content files are not directly accessible.

```python
import os, sys
sys.path.insert(0, os.environ["PROBLEM_DIR"])
from reader import load_batch
samples, inputs, labels = load_batch(batch_path)  # Path returned by fetch_data
i = 0
first_table = inputs[i][0]
if samples[i]["labeled"]:
    target = labels[i]
```

`load_batch` returns `(samples, inputs, labels)`:

- `samples[i]`: ID, ordinal, grouping, and authorization metadata for sample i.
- `inputs[i][j]`: input table j for sample i, represented as a dictionary mapping field names to one-dimensional arrays; see above for the meaning of each table.
- `labels`: a numeric array of shape `(n_batch,1)`; unauthorized samples use NaN placeholders. Determine authorization status from `samples[i]["labeled"]`.

A batch on disk contains `samples.json`, `input_j.npz`, and `labels.npy`. Each input table is stored using keys such as columns, offsets, and column_0; offsets delimit the rows for each sample, preserving variable lengths. Use the reader directly without parsing the packed layout yourself.

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | group_id = Country_code + normalized Site_name; holdout_group uses the same water-body group. Grouping by name does not additionally guarantee that real-world water bodies have undergone geographic entity disambiguation. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

