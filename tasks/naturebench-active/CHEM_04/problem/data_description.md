# Evidence schema and evaluation

Keep the three target-specific corpora separate. Pool inputs contain only the native target condition; an acquisition reveals one corpus SMILES. The shared native prior model is an allowed resource. There is no official Val or fixed Test sequence set: Task selects validation from acquired corpus records and uses model loss for selection; final inference generates exactly 500 SMILES per target. Native Success Rate and Scaffold Diversity score generated molecules after research closes. RDKit property calculations during development are allowed and do not reveal unqueried corpus records.

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

Development metric: model_loss. Validation origin:
acquired_labels. Instances requiring Task-selected validation:
qed, tanimoto_mol_a, tpsa.
The submission contract supplied by the host specifies fit/predict and, for
loss-based validation, validation_loss. Final native Test scores do not enter
Task/Meta research history. Scientific context below describes native Test metrics;
it does not override this package's development validation protocol.
