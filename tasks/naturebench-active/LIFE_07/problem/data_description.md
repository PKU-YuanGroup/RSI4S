# Input and label contract: LIFE_07

Inputs contain nucleic-acid sequences and a density map made available by host fetch. The query returns reference_mmcif. The two Test instances share one training-map pool. Pairing uses local mmCIF EMDB cross-references: 58 maps with multiple candidate structures are listed in organizer/source_exclusions.json and excluded until their pairing can be resolved; no arbitrary reference is selected. Density identity is computed from numeric MRC grid metadata and decompressed payload, not gzip filenames or timestamps. Exported MRC annotation labels are cleared, while all voxels and numeric grid fields are preserved. Research never receives a reference CIF for an unacquired record.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: shared_maps.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
