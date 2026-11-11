# Execution instructions: ENV_03

{{MEDRSEE_BATCH_READER}}

## ENV_03 inputs and predictions

Each sample's input table contains 24 forecast hours. Labels are returned
separately by the reader. Preserve selector order, including the original
`24:00` timestamp convention and missing target hours. Return finite
`values.npy` with shape `(N,24,99)`, quantiles `0.01..0.99`, and global `ordinal`
order. Internal validation separates dates, keeping all wind farms on a date
together.

See README.md and data_description.md for the full scientific and data contract.
