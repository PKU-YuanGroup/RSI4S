# Execution instructions: MED_04

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_04 inputs and predictions

Read CT `input_0` in `(X,Y,Z)` order and its `affine_0`. For acquired scans,
`samples[i]['annotations']` contains `[included, excluded]` nodule annotations;
no voxel segmentation mask is returned. Convert candidate voxel coordinates
through the native affine to physical millimeters. Return `prediction_0` with
shape `(K,4)`, at most 100 candidates, columns physical X/Y/Z and confidence;
an empty result has shape `(0,4)`. Keep each scan intact for internal validation.

See README.md and data_description.md for the full scientific and data contract.
