# Evidence schema and evaluation

Inputs: sequence. nrc targets are class_label (13 native class names); archiveii600/ts0 targets are partners, one 1-based partner index per nucleotide (0 means unpaired; -1 is a native unknown/unpaired sentinel). Nonstandard BPSEQ position numbering is remapped bijectively to sequence order; native_positions preserves the inverse. Structure instances retain their native V; nrc requires Task-selected validation from acquired labels. Missing list entries are recorded privately; existing entries and cross-split sequence duplicates are retained. Shared structure-instance rows are native task records, not independent biological observations. Development and Test use native macro classification F1 or mean per-structure base-pair F1.

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
nrc.
The submission contract supplied by the host specifies fit/predict and, for
loss-based validation, validation_loss. Final native Test scores do not enter
Task/Meta research history. Scientific context below describes native Test metrics;
it does not override this package's development validation protocol.
