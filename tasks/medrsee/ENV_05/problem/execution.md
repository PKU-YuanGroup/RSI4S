# Execution instructions: ENV_05

{{MEDRSEE_BATCH_READER}}

## ENV_05 inputs and predictions

Each sample's input contains a variable-length relaxation curve and the
published battery-condition fields. Labels are returned separately by the
reader. The capacity target is not an input column. Return `values.npy` with
shape `(N,1)` in Ah and global `ordinal` order. Internal validation isolates
complete cells; preserve the source sample definitions when constructing features.

See README.md and data_description.md for the full scientific and data contract.
