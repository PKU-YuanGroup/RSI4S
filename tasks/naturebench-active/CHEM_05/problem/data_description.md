# Input and label contract: CHEM_05

Small molecules expose only smiles, with their native scalar target. Protein instances expose primary sequence and protein_length, with log_fluorescence or stability_score as the acquired target. Source identifiers/topology/parent lookup metadata are omitted. Reactions expose the four Buchwald-Hartwig component SMILES or the Suzuki reaction SMILES; labels are Output or y. A single native reaction split is fixed per package. Ten alternative packages cover all native split choices; only one is mounted in a run, and package changes cannot resume an existing evidence ledger. Reaction instances have no V; the five other instances retain their cleaned native V.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: buchwald_hartwig, suzuki.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
