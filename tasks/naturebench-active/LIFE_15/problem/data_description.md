# Input and label contract: LIFE_15

Inputs contain native index, ID, sequence, and the cell instance's length/Gene fields. The acquired target is human, mouse, or KYSE-70. The other 322 cell-line label columns are omitted from the research package. Exact sequence identity spans assays for holdout grouping; assay-specific targets remain separate. Native files do not provide a common gene mapping for every human/mouse protein identifier, so exact-sequence grouping does not establish gene-family or isoform independence. Pretrained model files are not copied by this preparation adapter.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: cell_kyse70, human, mouse.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
