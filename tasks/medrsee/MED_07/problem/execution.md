# Execution instructions: MED_07

{{MEDRSEE_BATCH_READER}}

{{MEDRSEE_IMAGE_READER}}

{{MEDRSEE_VOLUME_PREFLIGHT}}

## MED_07 inputs and predictions

Read native CT `input_0`, `affine_0`, and the acquired `inputs['label_0']` mask.
Return `prediction_0` in the original spatial grid with `affine_0`. Preserve the
original vertebral identity IDs (`0` background, foreground within `1..28`);
do not collapse them to one foreground class or renumber by appearance. Use the
existing official validation set.

See README.md and data_description.md for the full scientific and data contract.

**MED_07 validation inference batch size.** Use `validation_batch_size=20` as the default and upper limit for MED_07 across Agent generations. This guidance applies only to MED_07 validation inference. Use a smaller batch when memory requirements demand it or for the final incomplete batch; do not increase it above 20 to use spare GPU memory. This is the number of examples passed to the model in one forward or prediction call, not the `fetch_data` page size or the number of validation members.

Process validation batches sequentially, without overlapping validation workers or keeping multiple batches on the GPU. Save each batch's predictions and release temporary tensors before processing the next batch. Evaluate every required validation member and preserve global `ordinal` order in the submission. The batch limit reduces resource contention but does not impose a GPU memory quota.
