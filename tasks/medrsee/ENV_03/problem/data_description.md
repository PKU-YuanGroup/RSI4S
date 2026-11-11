# ENV_03 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One wind-farm day, consisting of a fixed set of 24 hourly records. sample_id contains the wind farm and date; group_id is the date, used to keep wind farms on the same day on the same side of the split.

Native training pool 7,000, D0 700, Q 6,300, official V 0, T 310. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`inputs[i][0]` is a dictionary of named columns with 24 rows.

| Field | Meaning | Representation |
|---|---|---|
| ZONEID | Wind farm identifier | 1–10 |
| TIMESTAMP | Original hourly record time | Original strings and order are preserved |
| U10, V10 | Two horizontal wind components at a height of 10 m | Provided on the published numeric scale |
| U100, V100 | Two horizontal wind components at a height of 100 m | Provided on the published numeric scale |

The original 24:00 entry is the 24th record of the current sample and must not be moved to the next sample through timestamp parsing. Actual timestamps and targets are paired row by row; do not reorganize samples based on assumed UTC conventions, time zones, or forecast issue times. The precise time convention and wind-component units still require confirmation from the original release documentation.

## 3. Labels and Transformations Already Applied

`labels[i,:]` contains 24 normalized power output values, `TARGETVAR`; this task defines them as output relative to the wind farm's rated capacity. Authorized samples may still contain source NaN values; do not remove the entire day. Unauthorized samples also use NaN placeholders, so authorization must be determined from `samples[i]["labeled"]`, not solely from whether the targets are NaN.

The fixed 24 weather rows and corresponding targets are read without reordering hours, imputing targets, or removing days with missing values. Training and scoring may use observed targets; submissions must still cover all 24 hours and 99 quantiles.

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
- `labels`: a numeric array of shape `(n_batch,24)`; unauthorized samples use NaN placeholders. Determine authorization status from `samples[i]["labeled"]`.

A batch on disk contains `samples.json`, `input_j.npz`, and `labels.npy`. Each input table is stored using keys such as columns, offsets, and column_0; offsets delimit the rows for each sample, preserving variable lengths. Use the reader directly without parsing the packed layout yourself.

## 5. Sample Identification, Authorization, and Grouping

| Field | Role in this task |
|---|---|
| sample_id | Fixed unique identifier; do not truncate or regenerate it |
| ordinal | Fixed ordinal across the entire task; evaluation outputs follow ascending ordinal within the selected set |
| group_id | group_id is the target date, and holdout_group uses the same date; all ten wind farms on the same day are on the same side. The training period is 2012-01-01 through 2013-11-30, and this project's T is December 2013. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Uses the cumulative history of GEFCom2014-W Task15 and its fixed test period, not the full 15-round competition.
