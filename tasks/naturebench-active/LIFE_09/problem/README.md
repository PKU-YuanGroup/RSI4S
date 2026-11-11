# LIFE_09: active-evidence research

Predict the native scientific targets described in [scientific_context.md](scientific_context.md).
Read [data_description.md](data_description.md) for this package's input and label schema.
Included instances: H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K9me3, H4K20me1, deepstarr, enhancers, enhancers_types, promoter_all, promoter_no_tata, promoter_tata, splice_sites_acceptors, splice_sites_all, splice_sites_donors. Other native instances are outside this package.

| Training pool N | Initially labeled D0 | Unlabeled Q | Additional annotation budget | Official Val | Test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 893421 | 89342 | 804079 | 446710 | 40570 | 80008 |

Task proposes evidence requests; Meta chooses which Q samples to label. The budget is shared
across all generations of one four-hour run. Unqueried labels remain hidden.
Use official Val where available. Otherwise register exact acquired sample IDs for internal validation. Choose your own splitting tools and revise or clear the selection during research.
Always read the current acquired/role fields for training and validation membership.
Test is absent during research and reserved for final frozen inference.

Use only this public problem directory and the host tools. The host supplies the model
submission interface and current budget separately. Scientific background and metrics come
from NatureBench; the masked data view and annotation protocol are RSI adaptations.
