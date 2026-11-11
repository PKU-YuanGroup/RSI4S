# Execution instructions: ENV_02

{{MEDRSEE_BATCH_READER}}

## ENV_02 inputs and predictions

Each sample's inputs contain its authorized weather-history and static-basin
tables. Labels are returned separately by the reader. Neighboring discharge
labels are not input features. Return `values.npy` with shape `(N,1)` in mm/day
and global `ordinal` order. Preserve the complete supplied input history for
each sample. Choose internal validation with your own splitting tools.

See README.md and data_description.md for the full scientific and data contract.

## ENV_02 validation and prepared Test inputs

Register the exact acquired validation sample IDs with select_validation.
You may revise or clear the selection during research. The host does not preview,
diagnose or automatically expand the selection. The original basin-macro NSE
scorer returns actual metric results or failures for the submitted cohort.

ENV_02 requires lossless Test input batches prepared in a private host cache
before research starts. Missing or stale inputs stop startup before Agent calls.
This preparation does not fit features or models and does not include Test
labels. The cache is inaccessible to Task and Meta; only after research closes
does the frozen inference process receive a read-only view. The final time limit
covers cache verification and view creation, frozen inference, and scoring. The
two modes reuse the same input cache; it provides no extra training or Test feedback.
