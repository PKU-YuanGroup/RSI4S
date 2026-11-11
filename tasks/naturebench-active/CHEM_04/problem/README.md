# CHEM_04: active evidence research

This package follows the task-specific review in nb_pending_tasks_Lh.md.
Read data_description.md for the actual development and terminal evaluation protocol.
All native instances are included: qed, tanimoto_mol_a, tpsa.

| Training pool N | Initial D0 | Hidden Q | Additional query budget | Preset Val | Final conditions or samples |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2376404 | 237640 | 2138764 | 1188202 | 0 | 3 |

Task requests evidence; Meta may approve, modify, reject or initiate a query.
All generations share one annotation ledger and 14400 seconds of research.
Use official Val where available. Otherwise register exact acquired sample IDs for internal validation. Choose your own splitting tools and revise or clear the selection during research.
Read acquired=1 AND role='train' for fitting and acquired=1 AND role='val' for validation.
Unqueried supervision stays hidden; Test is released only after model selection and freeze.
Native row overlaps and augmentations are retained as instructed by the review.
These are record counts and do not imply independent biological observations.
