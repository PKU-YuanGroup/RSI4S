# Execution instructions: MED_08

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

## MED_08 inputs and predictions

Read native `(1,H,W)` grayscale ultrasound input and acquired
`inputs['label_0']` masks. Return `(H,W)` `prediction_0` with classes `0/1`,
restoring each image's original dimensions after any chosen resizing. Internal
validation keeps complete patient groups together.

See README.md and data_description.md for the full scientific and data contract.
