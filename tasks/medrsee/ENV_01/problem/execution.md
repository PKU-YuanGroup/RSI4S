# Execution instructions: ENV_01

{{MEDRSEE_BATCH_READER}}

## ENV_01 inputs and predictions

Each sample's inputs are a list of named-column table dictionaries; use the
spectral columns and repeated-observation selectors described in
`data_description.md`. Labels are returned separately by the reader. Handle
missing spectra using the chosen training procedure. Return `values.npy` with
shape `(N,1)` in `log10(Chla)` units and global `ordinal` order. Internal validation
isolates complete waterbody groups.

See README.md and data_description.md for the full scientific and data contract.
