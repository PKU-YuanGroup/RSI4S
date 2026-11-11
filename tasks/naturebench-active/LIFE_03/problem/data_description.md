# Input and label contract: LIFE_03

This task covers all eight native Test instances. PPI inputs are query and text amino-acid sequences; mutation_effect inputs are wild_seq, mutant_seq and participant_sequence. Acquired labels are {"label": 0 or 1}; predictions must be {"prediction_score": probability in [0,1]}. Mutation probability means increasing interaction strength. Native mutant_seq strings can contain dot placeholders; they are preserved without imputation, and the model chooses its sequence encoding. Feature type, Feature_type and all other native outcome/annotation metadata are omitted. The five cross_species_* instances share byte-identical human train/Val copies, represented once under cross_species_mouse. Train one shared human PPI model and apply it to all five species at final inference; mutation models may transfer from this acquired-label-trained PPI model as described in the native task. No externally trained PPI checkpoint or auxiliary labels are supplied. Bernett uses its full training CSV, not an additional copy of its 80% subset. PPI identities are global unordered exact sequence pairs; mutation identities are ordered full sequence triplets. Development groups touching any Test input are excluded before sampling; official Val takes priority over training, ambiguous development targets are quarantined and identical training targets deduplicated. Exact grouping does not establish sequence-homology independence. Native Test rows, duplicates and order remain unchanged. Official shared human, Bernett and mutation Val are retained after overlap checks. Only virus_human requires Agent-selected validation from acquired labels. Shared human Val is reported once under cross_species_mouse and cannot measure cross-species generalization. AUPR uses the native average_precision_score, not trapezoidal PR area; mutation also reports AUROC. The host writes output/{instance}/predictions.csv in native Test order only after final selection and research closure.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: virus_human.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
