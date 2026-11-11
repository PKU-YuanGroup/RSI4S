# MED_02 Data Description

This file describes the BraTS2021 data the Agent receives. See [README.md](README.md) for the scientific objective, budget, validation, and submission rules.

## 1. What Constitutes a Sample

One case contains four three-dimensional MRI modalities and one segmentation mask. The training pool has 1000 cases, D0=100, Q=900, official V=0, and project T=251. `sample_id`, `group_id`, and `holdout_group` all identify the fixed case. Preprocessing or training crops do not change the query unit.

## 2. Input Fields and Shapes

| Returned location | Content | Shape / type |
|---|---|---|
| `inputs[i]["input_0"]` | MRI stacked in T1, T1ce, T2, FLAIR order | (4,X,Y,Z), float32 |
| `inputs[i]["affine_0"]` | Native voxel-to-physical-coordinate transform | (4,4) |
| `inputs[i]["label_0"]` | Segmentation mask for an authorized case | (X,Y,Z), int16; absent when unauthorized |

Spatial axes retain the NIfTI array order, and all four modalities share a grid. Released images are typically 240×240×155 with 1 mm isotropic spacing; use the returned arrays and source headers for the actual geometry. No separate spacing key is returned. MRI intensities retain their published scale and are not treated as absolute quantitative physical measurements.

## 3. Labels and Transformations Already Applied

Labels are 0 background, 1 necrotic tumor core, 2 edema, and 4 enhancing tumor. The reader does not remap 4 to 3. If the model uses internal recoding, restore the original codes before submission.

The organizer reads NIfTI files at fixed offsets in the original TAR archive, decodes the four modalities, casts them to float32, stacks them, and checks grid agreement between images and authorized masks. Masks are cast to int16, and the source affine is retained. No automatic normalization, cropping, resampling, or augmentation is applied. Model preprocessing is chosen by the Agent.

## 4. How to Read a Batch

Obtain a batch path through `fetch_data`, then use the framework-provided `PROBLEM_DIR/reader.py`:

```python
import os, sys
sys.path.insert(0, os.environ["PROBLEM_DIR"])
from reader import load_batch
samples, inputs, labels = load_batch(batch_path)
i = 0
item = inputs[i]
x = item["input_0"]
affine = item["affine_0"]
if samples[i]["labeled"]:
    y = item["label_0"]
```

`samples[i]` is a metadata dictionary containing sample_id, group_id, split, ordinal, holdout_group, role, and labeled; the reader handles file locations. `inputs` is a lazy sequence indexed by sample, and `inputs[i]` is a dictionary of arrays. The third return value, `labels`, is `None`; masks are in `label_0` above. Explicitly copy arrays before in-place preprocessing.

Fetch a new batch after authorization to obtain new labels; old batches do not update automatically. An unauthorized sample has no label_0, which does not mean all background. Reading all four modalities together can use substantial memory, so start with one case. When using the shared three-dimensional resizing helpers, process each channel's spatial three-dimensional array separately.

## 5. Membership and Authorization

During research, the membership catalog is `DATA_DIR/pool.sqlite`; select sample IDs before fetching data. The catalog uses `acquired` for label authorization, while batches use `labeled` to indicate whether labels are actually included. Training members must be authorized and have `role="train"`; queried Q members retain their original split="Q". `ordinal` is the fixed order, and submissions follow ascending ordinal within the evaluated set.

The organizer's original archive, split CSV containing label locations, and stratification statistics are not exposed to the Agent. WT/ET stratification fields are not inputs. At the final stage, `DATA_DIR/batches.json` lists batches containing only T inputs, read with the same reader; no T labels are provided.

## 6. Source

The source is a pinned third-party public mirror, with archive `BraTS2021_Training_Data.tar`. The fixed split version is `custom1000_251_v1`; official unlabeled V and hidden T are not included in this mirror. Migration preserves the same archive and actual members without generating a new split.
