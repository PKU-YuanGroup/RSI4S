# Execution instructions: MED_05

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

## MED_05 inputs and predictions

Read the native `(3,256,256)` RGB input and the acquired `inputs['label_0']`
mask. Its two target channels are instance ID and cell type. Return integer
`prediction_0` with shape `(256,256,2)`, preserving consistent type assignments
within each instance. Do not treat instance IDs as semantic classes. Internal
validation isolates the published source-image groups.

See README.md and data_description.md for the full scientific and data contract.
