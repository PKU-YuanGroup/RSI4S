## Authorized batch access

`fetch_data` returns read-only native batches. Use `limit` (1–4096 samples per
page) and optional `pages` (1–256) to fetch several pages in one host call.
Completed batches are recorded in `DATA_DIR/batches.json`; reuse them instead of
fetching the same evidence repeatedly. A partial multi-page fetch reports its
completed batches and `next_offset` if the stage deadline interrupts it.

Import `load_batch`, `iter_batch` or `iter_dataset` from `PROBLEM_DIR/reader.py`.
`iter_batch(path, role=selected_role)` yields `(sample, inputs, label)` and filters
sample metadata before reading the selected inputs. The task's data description
defines the input representation and where acquired labels are returned.

The reader resolves current roles from the public `DATA_DIR/pool.sqlite`, even
for batches fetched before validation was declared. Raw `samples.json` stores
export-time roles; do not use that snapshot or an old cached role list to describe
current membership. Reload with the reader after registering validation. Keep
batch directories in their original data tree so the membership database is available.

To combine overlapping exports, use
`reader.iter_dataset(DATA_DIR, role=selected_role, labeled_only=True)`. Choose
`selected_role` for the current experimental design (`None` disables role
filtering). It yields each sample ID once in ascending native ordinal order,
preferring an authorized labeled export and then the latest export. It does not
fetch additional data or reveal unqueried labels. An old batch does not gain
labels after a query; fetch approved samples again for a new authorized view.

During verified validation and final inference, read only the batches supplied
in that run's `DATA_DIR/batches.json`; they have `labeled=False`:

```python
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ['PROBLEM_DIR'])
from reader import load_batch

data_dir = Path(os.environ['DATA_DIR'])
output_dir = Path(os.environ['OUTPUT_DIR'])
for batch in json.loads((data_dir / 'batches.json').read_text()):
    samples, inputs, labels = load_batch(batch['path'])
    # Load saved parameters, infer from inputs, and save task-specific predictions.
```
