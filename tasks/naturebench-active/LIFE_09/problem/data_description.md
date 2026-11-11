# Input and label contract: LIFE_09

Inputs contain DNA sequence and, for human classification assays, a genomic region without the native |label suffix. DeepSTARR FASTA headers contain activity-related flags, so they are omitted. Labels are label for classification, or the two raw Dev_log2_enrichment/Hk_log2_enrichment values. Derived scaled/quantile targets are not input. Holdout groups connect exact/reverse-complement sequences and overlapping intervals within the same species/chromosome, including relationships across assays. DeepSTARR retains native V; the 18 classification instances require Task-selected validation from acquired labels. This is a mixed-validation package.

Read data/samples.sqlite using Python sqlite3, or use fetch_data for paginated records.
The samples table has sample_id, instance, ordinal, group_id, split, acquired,
role, inputs and label. Inputs and labels are JSON; an unqueried Q label is SQL NULL.
The assets table contains input payloads only. Numerical missing values are JSON null.
Use returned attachment paths; original archives and organizer files are not research inputs.

One query reveals all defined targets for that record. New unique records consume the shared
annotation budget; duplicate aliases and excluded records add no quota. Use annotation_status
for the current balance. Native target names, units, and Test row order are preserved.

Instances requiring Task-selected validation: H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K9me3, H4K20me1, enhancers, enhancers_types, promoter_all, promoter_no_tata, promoter_tata, splice_sites_acceptors, splice_sites_all, splice_sites_donors.
Validation declarations use the exact acquired sample IDs submitted by Task.
Each call replaces internal membership; [] clears it. Choose your own splitting tools.
Read acquired=1 AND role='train' for training and acquired=1 AND role='val' for validation.
Newly acquired labels are not automatically added to internal validation.

Development deduplication and exclusions are recorded in the package manifest. This active
evidence protocol can change the native training population; native descriptions may report
pre-curation counts.
