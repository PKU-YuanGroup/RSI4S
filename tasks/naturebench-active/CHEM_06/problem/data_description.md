# Evidence schema and evaluation

Inputs: num_residues and one input-only NPZ containing pkt_node_xyz, pkt_node_sca, pkt_node_vec, pkt_seq, ligand_center_of_mass. The center is a native conditioning input. Acquired targets: ligand_element, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_atom_feature, ligand_nbh_list. A fixed seed-42 10% holdout of available native training entries is V; the other 90% defines N. Development uses the model's loss. Final inference generates 100 SMILES for each of the original 91 pockets; native Test scoring reports Vina Score, QED, SA and Diversity. Vina is not a development metric.

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
review_instructed_fixed_holdout. Instances requiring Task-selected validation:
none.
The submission contract supplied by the host specifies fit/predict and, for
loss-based validation, validation_loss. Final native Test scores do not enter
Task/Meta research history. Scientific context below describes native Test metrics;
it does not override this package's development validation protocol.
