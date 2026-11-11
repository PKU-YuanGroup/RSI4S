# ENV_05 Data Description

This file describes the data the Agent actually receives. See [README.md](README.md) for the scientific objective, query budget, validation rules, evaluation, and prediction submission.

## 1. What Constitutes a Sample

One continuous diagnostic cycle segment of one cell. sample_id contains the cell, cycle number, and run; a restarted cycle number does not mean the sample is the same as an earlier cycle.

Native training pool 53,823, D0 5,382, Q 48,441, official V 0, T 5,110. Fixed membership does not change with preprocessing, missing-data handling, or cropping for training.

## 2. Input Fields and Shapes

`inputs[i][0]` is a relaxation sequence of variable length m, with operating-condition attributes repeated as constant columns in every row.

| Field | Meaning | Unit / encoding |
|---|---|---|
| time/s | Time relative to the first point of the selected segment | s |
| Ecell/V | Relaxation voltage | V |
| temperature_C | Published operating temperature | °C |
| chemistry | Chemistry | NCA, NCM, NCM_NCA |
| nominal_capacity_Ah | Nominal capacity | Ah |
| charge_rate_C | Extracted charging rate | C-rate; provided when determinable, subject to the columns actually returned |

The discharge-capacity column is not exposed as an input, nor are all original current/control columns used for organizer-side selection. The same number of points, sampling interval, or duration is not guaranteed across chemistries and samples.

## 3. Labels and Transformations Already Applied

`labels[i,0]` is the maximum `Q discharge/mA.h` in the same continuous diagnostic interval multiplied by 0.001, in Ah; this maximum must occur after the selected relaxation segment. The quantity estimated here is current capacity, not remaining life.

A new continuous segment starts when the source cycle number changes or time moves backward. Within a segment, select the first continuous relaxation interval satisfying control state 0, |I|≤1 mA, at least 10 finite sample points, all voltages >4 V, and strictly increasing time. Return the entire interval; subtract the first time value without resampling onto a common grid, and convert capacity from mAh to Ah. These rules have already been used to establish fixed membership; the Agent is not required to select or remove samples again.

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
| group_id | Both group_id and holdout_group identify the cell; different cycles of the same cell share a group, but labels are still authorized independently for each sample. The outer split holds out cells within each represented chemistry. |
| holdout_group | Group used to separate validation; see above for its meaning and README for the separation rules |
| split | Original D0, Q, or V membership; Q membership is retained after querying |
| acquired / labeled | The membership catalog uses acquired; batches use labeled to indicate whether labels are actually included in that batch |
| role | train or val in the membership catalog; training labels must both be authorized and belong to train |

During research, the membership catalog is `DATA_DIR/pool.sqlite`; filter by sample_id or group before fetching data. `fetch_data` supports specified IDs or pagination, with at most 4096 samples per batch; large volumes should be read in small batches. Fetch data again after acquiring new labels; old batches do not update automatically.

This task has no official V. Initial D0 labels are available, while Q requires authorization; declare internal validation roles as described in README. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read using the same reader; there is no training membership catalog or T labels.

## 6. Sources

Segment selection and pairing follow this project's fixed selector.
