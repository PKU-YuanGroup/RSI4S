# Execution instructions: ENV_04

{{MEDRSEE_BATCH_READER}}

## ENV_04 inputs and predictions

Each sample's inputs contain its daily and static-site tables. Labels are
returned separately by the reader. Fit missing-value or sentinel handling on
authorized training data. Return `values.npy` with shape `(N,1)` in
`gC m^-2 day^-1` and global `ordinal` order. Internal validation isolates the
published site-neighborhood groups; score aggregation remains by site.

See README.md and data_description.md for the full scientific and data contract.
