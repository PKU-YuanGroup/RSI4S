# Evidence schema and evaluation

Inputs: src_smi, src_atom_n, task_name, and admet re_group. The acquired label contains tgt_smi and available target opt_pred_* supervision. Native training and validation rows (including repeated source molecules and task conditions) are retained. The standard ADMET corpus is used; its few-shot superset and auxiliary property-labeled corpora are not free evidence. Validation generates candidates and uses the native property oracles and success criteria. Test requires 20 candidates per drd2_qed source and 5 per admet source.

The data/samples.sqlite samples table contains sample_id, instance, ordinal,
group_id, split, acquired, role, inputs and label. Inputs/labels are JSON; every
unqueried Q label is SQL NULL. Never train on V or unqueried Q. Selecting validation
changes roles and does not acquire labels. The shared budget is floor(N/2), in
addition to floor(N/10) initial records. IDs, ordinals and hashes are bookkeeping,
not learned scientific features.

Use fetch_data to resolve sanitized feature attachments. Model workers receive
absolute inputs.feature_paths for NPZ inputs (np.load with allow_pickle=False),
and inputs.resource_paths for allowed shared resources, when present. Only these
explicit file paths are permitted. Original archives, full labeled corpora and
organizer files are not model inputs. Resources in reference/ are public.

Development metric: native. Validation origin:
official. Instances requiring Task-selected validation:
none.
The submission contract supplied by the host specifies fit/predict and, for
loss-based validation, validation_loss. Final native Test scores do not enter
Task/Meta research history. Scientific context below describes native Test metrics;
it does not override this package's development validation protocol.
