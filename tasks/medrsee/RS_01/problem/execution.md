# Execution instructions: RS_01

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

## RS_01 inputs and predictions

Use the released `(14,128,128)` image array and acquired `inputs['label_0']`
masks. Return native `(128,128)` `prediction_0` with classes `0/1`. Preserve the
existing official Val. The primary score pools pixel counts across images;
follow the published scoring contract.

See README.md and data_description.md for the full scientific and data contract.
