# Input and label contract: LIFE_10

Inputs expose variant and num_mutations; score is the acquired assay target. Shared reference files contain parent protein structures, never variant-effect labels. Holdout groups preserve exact variant identity within each parent assay. Additional mutation/site holdout is an Agent research choice; the native Test evaluates the original within-assay distribution.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: dlg4_abundance, dlg4_binding, gb1, gfp, grb2_abundance, grb2_binding, pab1, pten_abundance, pten_activity, tem_1, ube4b.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
