## Native 3-D execution preflight

Before cohort-scale preprocessing or training, test one acquired training case
and one small training step. Record input/output shapes, dtype, finite values,
target identities, elapsed time, RSS and CUDA peak allocation; repair failures first.
Native volume exports may be read-only NPY memory maps: `samples.json` uses
`arrays` instead of `file` for these exports. NPZ batches remain supported;
use the public reader for either representation.

`reader.volume_resize_plan(shape, dtype, target_shape)` checks three spatial
dimensions and reports output-array bytes before allocation; model activations
require extra memory. `resize_volume(x, target_shape, order=1)` rejects enlargement
with its default `intent="downsample"`. For multichannel volumes, apply these
helpers to each spatial volume, keeping channel and batch axes separate.

`downsample_volume(x, (4,4,2))` treats factors as divisors, producing approximately
1/32 as many voxels. Passing `(4,4,2)` directly to `scipy.ndimage.zoom` enlarges the
volume 32-fold instead; zoom expects output/input ratios. For discrete label masks,
use `order=0` and preserve label identities. If predictions are spatial arrays,
restore their original grid with explicit native dimensions and `intent="resize"`,
and preserve the native affine. Detection coordinates follow the task's own
physical-coordinate output contract. These helpers do not choose a scientific model.
