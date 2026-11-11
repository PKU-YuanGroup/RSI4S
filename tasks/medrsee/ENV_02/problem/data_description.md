# ENV_02 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One target day for one catchment. sample_id ends with the target date in YYYY-MM-DD format, and group_id is the catchment identifier.

Native training pool 7,347,592, D0 734,759, Q 6,612,833, official V 0, T 2,409,999. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`inputs[i][0]` is a meteorological window of 366 rows, covering the 365 days before the target day and the target day, returned in date order. `inputs[i][1]` is one row of static attributes for that catchment.

| Table | Field | Meaning | Unit |
|---|---|---|---|
| Meteorology | date | Date of the corresponding row | YYYY-MM-DD |
| Meteorology | precipitation | Catchment-mean daily precipitation | mm/day |
| Meteorology | pet | Potential evapotranspiration for reference grass | mm/day |
| Meteorology | peti | Potential evapotranspiration with a rainfall interception correction | mm/day |
| Meteorology | temperature | Daily mean air temperature | °C |
| Meteorology | humidity | Daily mean specific humidity, not relative humidity | g/kg |
| Meteorology | shortwave_rad | Daily mean downward shortwave radiation | W/m² |
| Meteorology | longwave_rad | Daily mean longwave radiation | W/m² |
| Meteorology | windspeed | Daily mean wind speed | m/s |
| Static | gauge_lat, gauge_lon | Gauging station latitude and longitude | Degrees |
| Static | gauge_elev | Gauging station elevation | m |
| Static | area | Catchment area | km² |
| Static | dpsbar | Mean slope of catchment drainage paths | m/km |
| Static | elev_mean | Mean catchment elevation | m |

Each field is a one-dimensional array: meteorological fields have length 366, and static fields have length 1. Target-day meteorological data are inputs; runoff from neighboring days is not.

## 3. Labels and Transformations Already Applied

`labels[i,0]` is target-day `discharge_spec`, in mm/day. One query authorizes only one target day, not runoff for other dates within the window.

The complete window is extracted at fixed row positions and joined with static attributes, without converting input units or fitting any scaling. Dates come from the source table, and the last row of the window is the target day. WY1972–2005 corresponds to 1971-10-01 through 2005-09-30; WY2006–2015 corresponds to 2005-10-01 through 2015-09-30. Actual valid membership still follows the fixed list.

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
| group_id | group_id identifies the catchment, and scores are aggregated by catchment; internal validation is separated by a target-date boundary, rather than by holding out entire catchments. The same catchment may span the training and test periods. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Field meanings and units have been checked against the source documentation.
