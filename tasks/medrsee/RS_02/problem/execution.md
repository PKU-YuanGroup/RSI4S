# Execution instructions: RS_02

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

## RS_02 inputs and predictions

Use the released 11-channel `(11,240,240)` multispectral array. Acquired masks
are `inputs['label_0']`; distinguish ignored truth pixels (`-1`) from input
missing values. Return `prediction_0` with native shape `(240,240)` and classes
`0..14`. Preserve the published channel order and use the existing official Val.

See README.md and data_description.md for the full scientific and data contract.
