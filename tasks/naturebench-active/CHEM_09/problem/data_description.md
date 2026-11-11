# Input and label contract: CHEM_09

A single shared pool contains the 500 native train_300K configurations, used for all four Test regimes. Inputs contain species, positions, lattice, and pbc. The query returns energy and per-atom forces together. The alternative mixed-temperature training pool and labeled isolated-atom reference file are not additional free labels. Available files do not identify independent trajectories; groups currently cover exact configurations only, not temporal decorrelation. No automatic V is created; an internal 300K holdout cannot establish high-temperature generalization.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: shared_300k.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
