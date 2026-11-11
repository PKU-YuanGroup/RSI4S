## Image batch execution

For images, `load_batch` returns a lazy input sequence: accessing one item loads
only that case. Use `iter_batch` to filter sample metadata before reading arrays.
Copy only arrays you intend to modify in place. The separate `labels` return is
`None`; use the task's specified mask or annotation fields for acquired labels.

Write one prediction NPZ per sample, numbered from `000000.npz` in global
ascending `ordinal` order across all batches. Include `sample_id` and the task's
required prediction arrays and affine fields. The task's data description gives
the exact shape and label convention; do not assume all image tasks use the same
target representation.
