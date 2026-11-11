# Input and label contract: LIFE_04

One retained training record is one gene annotation shared by the eight graph networks. Identical gene symbols have a common identity across networks. The retained record's instance/network is a canonical storage location: acquired labels may be mapped by gene symbol to every eligible development graph containing that gene. Initial D0 does not guarantee two classes within each storage-instance subset. A gene held out in any network cannot supply a training label through another network. Exact repeated training labels are represented once; native Test rows remain separate and in order. Inputs identify gene, network, native node_index, and graph_asset. Label is cancer_gene. The assets table stores node indices, names, 64 features, and edge_index in native node-index space. Development graphs exclude every gene held out for Test anywhere. Final assets contain the full input graphs and no labels. This explicitly changes the native transductive development view to enforce terminal-only Test input access; results must identify this protocol.

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
