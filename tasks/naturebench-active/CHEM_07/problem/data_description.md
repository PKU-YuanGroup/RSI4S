# Input and label contract: CHEM_07

Inputs contain fragments, fragment_molblock (3D SDF mol block with empty metadata), and anchors. Pockets additionally contain full_coord/full_types and bb_coord/bb_types. A query reveals the native molecule/linker SMILES and molecule_molblock/linker_molblock. Full-molecule names, conformer energies, linker masks, total molecule sizes, and other target-derived table fields are absent from inputs. Open aromatic fragments retain their native aromatic bond representation without forcing kekulization. ZINC training and validation are shared once with CASF; the three pocket conditioning variants share one full-pocket evidence record. Exact fragment geometry defines duplicate identity; fragment connectivity, plus pocket identity where provided, defines holdout grouping. This does not prove that unknown Test full molecules are distinct from all development molecules.

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
