# LIFE_01 input and label contract

The prepared research package exposes data/samples.sqlite. Its samples table
contains sample_id, instance, ordinal, group_id, split, acquired, role, inputs,
and label. Inputs and labels are JSON. The only instance is atezolizumab.

Inputs:
- sequence: the amino-acid sequence.
- chothia_sequence: the same sequence on the original 132-position Chothia grid;
  '-' denotes an unused position. Alphabet order is ACDEFGHIKLMNPQRSTVWY-.
- embedding: the flattened 95 by 3 task-adapted autoencoder representation
  (285 floats, positions 29:124 in the original grid). The encoder was pretrained
  on independent human B-cell receptor sequences and artificial decoys; these
  features do not use this assay's binding labels.

Label (only for acquired records):
- class: integer 0=RH01, 1=RH02, 2=RH03.
- confidence_weight: (maximum bin count + 1)/(total bin count + 3).

Unqueried labels are SQL NULL,
including their confidence weights. Only the fields listed above are public inputs.

Sample IDs preserve original row order; group IDs identify exact full sequences.
All train/test sequences must be unique and disjoint. IDs are alignment keys,
not model features. Read acquired=1 AND role='train' for fitting and
acquired=1 AND role='val' for validation. A validation declaration moves the
entire group, but never reveals any unqueried labels.

The public template contains D0 and Q only. Test labels are not placed in the
active reference database or any research view. During final frozen inference
the host releases Test inputs without any label field. Predictions are serialized
as atezolizumab/predictions.csv with sample_id,p_RH01,p_RH02,p_RH03,predicted_class.
