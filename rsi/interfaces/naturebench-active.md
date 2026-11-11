# NatureBench active-evidence submission: {{TASK_ID}}

Use the curated active package and its acquired-label ledger. The Agent decides
how to compare models and when to create, expand or revise internal validation.
Test inputs are absent during research. Task may request labels;
only Meta may acquire Q labels. All generations share the configured total research budget.
The submission contract exposes `evaluation_phases: {"policy":"shared_deadline_v1"}`.
Fitting, validation and scoring each receive all time remaining before the current
evaluation/stage deadline. There are no phase quotas, reservations or separate
phase caps. Each evaluation saves actual durations and deadlines in its feedback
and `phase_timing.json`.
Read annotation_status first. Official V remains available where supplied.
For instances without V, Task may call select_validation with exact acquired
D0 or Q sample IDs. Each call replaces the internal selection; include retained
and new IDs to expand it, or pass [] to clear it. The Agent chooses or writes
its own splitting tools. The host does not expand groups or perform validation
preflight. New Q labels remain training members until explicitly selected.
Scoring reports actual metric results and failures. Evaluation and final selection
may use a subset of instances; feedback records evaluated and unevaluated instances.
Past evaluations retain their actual memberships and scores. When comparing rounds,
account for changes in the model, training data and validation samples.
Official validation and Test boundaries remain unchanged.
Read current sample roles after declarations or queries. The host evaluation uses
acquired role='train' members for fitting and acquired role='val' members for
validation. Label local evaluations with their actual membership.

Save a self-contained `model.py` defining `fit(task, records)` and
`predict(model, records)`. The host `evaluate` operation fits it on ALL acquired
training-role records in an isolated worker with the experiment's assigned GPU
visible, saves the returned pickleable
model, then runs prediction in a different worker that can read only unlabeled
validation inputs and that fitted model. Predictions are one dictionary per input,
in input order. Training records contain sample_id, instance, group_id, inputs,
label; prediction records omit label. IDs are alignment keys, not features.
Choose your own internal CV design. Do not read external files or network data.
Only explicitly supplied feature_paths and resource_paths are permitted
attachment reads. The task-specific submission contract takes precedence:
tasks with model-loss validation define fit(task, training_records,
validation_records) and validation_loss(model, labeled_records). Their validation
records may be used for epoch/hyperparameter selection and early stopping only;
the host evaluates the supplied loss using a frozen fitted model. Self-reported
losses are diagnostics: they have no cross-method aggregate improvement or
automatic best-attempt ranking. Explain the formula, units, normalization and
relation to the submitted generator when selecting a model. Loss evaluation does
not run a generation probe; the Agent chooses whether and when to test generation
during development. Their final predict receives
unlabeled generation conditions and must return generated candidates.

The public data file is `problem/data/samples.sqlite`, not pool.sqlite or a
MedRSEE batch reader. `fetch_data` returns JSON records and optional sanitized
attachment paths, one page at a time. Numeric arrays live in JSON inputs or the
explicit input attachments. Evaluation phases share the current research/stage
deadline under the supplied evaluation_phases settings. Use the assigned CPU
affinity and thread budget; the scientific worker preserves that allocation.
For neural models, move the model and input tensors to `cuda:0`, the logical name
of the assigned GPU inside the worker. GPU availability alone does not move a
model to CUDA. CPU models such as scikit-learn estimators remain supported.
Validation returns native metrics for the available development cohorts;
aggregate fields use a fixed unit scale, not any published value, and must not be
described as a measured Test improvement.

Task artifacts contain the submitted program; the host also saves its exact fitted
parameters, input memberships, predictions and native metric results. Meta may
change the successor Agent and approve new evidence. After research closes, Meta
must explicitly select an eligible validated artifact, explaining its choice.
Final inference uses its exact host-fitted model and releases unlabeled Test once.
The frozen Test predict run {{FINAL_TIMING}} and receives every
Test input, which can be many times more records than a validation cohort.
This terminal execution is outside the research and M-chain budgets. When untimed,
it runs until completion, an execution error or an explicit interruption.
It computes no Test score and supplies no feedback to subsequent research.
