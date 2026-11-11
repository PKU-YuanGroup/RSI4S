# Execution instructions: MED_01

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_01 inputs and predictions

Read ED and ES as separate native spatial volumes with their own affine
matrices. Acquired masks are `inputs['label_0']` and `inputs['label_1']`. Return
`prediction_0` for ED and `prediction_1` for ES in their respective native grids,
with `affine_0` and `affine_1`, preserving label IDs `0/1/2/3`. Internal validation
keeps both phases and all slices of a patient together.

See README.md and data_description.md for the full scientific and data contract.
