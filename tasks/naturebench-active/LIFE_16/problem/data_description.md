# Input and label contract: LIFE_16

Four native distance bins are gtex_v8/small, middle, large, huge. One query reveals label (direction classification) and slope (effect regression) together. pval_nominal and slope_se are never inputs. Public metadata contains gene/variant IDs for alignment, tissue bulk, distance, maf and allele counts; IDs and hashes are not learned features. Full reference/mutant sequences and both ATAC profiles are preserved losslessly in data/files/<hash>/features.npz. Each file contains four uint8 ASCII sequence arrays and two float64 ATAC arrays, without pickle objects or labels. fetch_data returns the file path, and host fit/predict supplies inputs.feature_paths. Group IDs connect shared Ensembl gene IDs (without version suffix) or chromosome/position across all tissues and bins, including alternative alleles at the same locus. Every development group touching Test is excluded. This changes the native within-cohort training protocol; it does not prove independence of all overlapping regulatory windows. There is no separate authoritative V membership file: the native description suggests a positional 8/9 split, which is not adopted as a fixed V. Task chooses grouped validation only from acquired labels. Predictions use predicted_label, predicted_probability, predicted_slope; the primary metric is the unweighted mean of four bin AUCs.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: gtex_v8/huge, gtex_v8/large, gtex_v8/middle, gtex_v8/small.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
