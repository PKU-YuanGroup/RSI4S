# ENV_04 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One valid day at one site. sample_id consists of the site and date; each sample has only one Obs target.

Native training pool 140,002, D0 14,000, Q 126,002, official V 0, T 16,025. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`inputs[i][0]` is one row of daily data; `inputs[i][1]` is one row of static site attributes. Each field is returned as an array of length 1.

| Table | Field | Meaning | Unit / encoding |
|---|---|---|---|
| Daily data | Unnamed: 0 | Site ID | String, such as AU-ASM |
| Daily data | Unnamed: 1 | Date | YYYY-MM-DD |
| Daily data | SW_IN_F_MDS | Downward shortwave radiation | W/m² |
| Daily data | TA_F_MDS | Air temperature | °C |
| Daily data | VPD_F_MDS | Vapor pressure deficit | Published scale; precise unit remains to be verified |
| Daily data | swvl1–swvl4 | Volumetric soil water content in four layers | m³/m³; layer depths in this release remain to be verified |
| Static | towerlat, towerlon | Site latitude and longitude | Degrees |
| Static | alt | Site elevation field | Source values are preserved; unit remains to be verified |
| Static | IGBP_Code | Vegetation functional type | Strings such as ENF, not integer identifiers |

Inputs contain only the fields in the table; other model GPP columns in the source data are not used as inputs.

## 3. Labels and Transformations Already Applied

`labels[i,0]` is `Obs`, in gC m⁻² day⁻¹. It is GPP obtained through daytime flux partitioning, sourced from GPP_DT_CUT_REF or GPP_DT_CUT_MEAN according to site quality. The reader does not apply additional scaling to Obs; this does not mean that every site uses the same REF product.

The fixed site-day and corresponding static row are read without additional unit conversion, normalization, or imputation. The selected source columns contain NaN values; −9999 was not found in these columns in this release, so −9999 should not be assumed to be the only missing-value marker. Determine imputation parameters on the permitted training side. VPD retains the published values; do not infer Pa or hPa as its unit from the numeric magnitude.

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
| group_id | group_id identifies the site for equally weighted site-level scoring; holdout_group is the connected group formed using 10 km proximity, used to separate training and validation. Sites within a group are connected through proximity relationships; not every pair of sites is required to be within 10 km. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

The source is a fixed data release.
