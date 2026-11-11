# Execution instructions: MED_02

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_02 inputs and predictions

Read four-modality `input_0` with shape `(4,X,Y,Z)` in T1, T1ce, T2 and FLAIR
order. Keep the modality axis separate when calling three-spatial-dimension
resize helpers. Acquired masks are `inputs['label_0']`. Return `prediction_0`
with native shape `(X,Y,Z)`, `affine_0`, and original label IDs `0/1/2/4`.
Internal validation keeps complete cases together.

See README.md and data_description.md for the full scientific and data contract.
