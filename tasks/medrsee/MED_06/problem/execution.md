# Execution instructions: MED_06

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_06 inputs and predictions

Read native ceT1 `input_0`, `affine_0`, and the acquired `inputs['label_0']`
mask. Return `prediction_0` in the original spatial grid with `affine_0` and
labels `0/1/2` for background, vestibular schwannoma and cochlea. Internal
validation keeps complete cases together.

See README.md and data_description.md for the full scientific and data contract.
