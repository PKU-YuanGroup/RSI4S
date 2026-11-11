# LIFE_13: active-evidence research

Predict the native scientific targets described in [scientific_context.md](scientific_context.md).
Read [data_description.md](data_description.md) for this package's input and label schema.
Included instances: cath_4_2, cath_4_3, pdb2022, ts50. Other native instances are outside this package.

| Training pool N | Initially labeled D0 | Unlabeled Q | Additional annotation budget | Official Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17723 | 1772 | 15951 | 8861 | 2057 | 5007 |

Task proposes evidence requests; Meta chooses which Q samples to label. The budget is shared
across all generations of one four-hour run. Unqueried labels remain hidden.
Use official Val where available. Otherwise register exact acquired sample IDs for internal validation. Choose your own splitting tools and revise or clear the selection during research.
Always read the current acquired/role fields for training and validation membership.
Test is absent during research and reserved for final frozen inference.

Use only this public problem directory and the host tools. The host supplies the model
submission interface and current budget separately. Scientific background and metrics come
from NatureBench; the masked data view and annotation protocol are RSI adaptations.
