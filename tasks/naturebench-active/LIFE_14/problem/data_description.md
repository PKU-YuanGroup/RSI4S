# Input and label contract: LIFE_14

Inputs preserve name, wt_seq, mut_seq, pH, and temperature where present. The acquired scalar target is ddG for s669 or dTm for s571, with its native sign. Different experimental conditions remain distinct records. Connected wild-type/mutant sequence components span both ddG and dTm assays, define validation groups, and cover reverse mutation relationships. Any component held out for Test in either assay is removed from both training pools. No claim of sequence-homology clustering is made.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: s571, s669.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
