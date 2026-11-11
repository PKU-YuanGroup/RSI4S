# Execution instructions: MED_03

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_03 inputs and predictions

Read native CT `input_0`, `affine_0`, and the acquired `inputs['label_0']` mask.
Return `prediction_0` in the original spatial grid with `affine_0` and labels
`0/1/2`. Preserve liver and tumor identities; internal validation keeps complete
cases together.

See README.md and data_description.md for the full scientific and data contract.
