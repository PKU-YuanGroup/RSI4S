# Input and label contract: LIFE_08

Inputs are JSON arrays projected from the original NPZ: V_prot, X_prot, E_prot, prot_vectors, V_dna, X_dna, X_dna_point, dna_vectors, dna_mask, prot_feature_names, dna_feature_names, contacts, Y_hard. The query returns Y_pwm and pwm_mask together. aln_score is omitted. Connected components of native protein clusters define holdout groups: clusters sharing any chain are joined. Chain letters retain their case. One official CV fold is fixed for each package; the other folds are separate alternatives and must never be visible to the same research run. Native folds had cluster overlap with Test; the corresponding development groups are removed.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: none.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
