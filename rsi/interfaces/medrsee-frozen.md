# Platform submission interface: MedRSEE

This interface applies to task `{{TASK_ID}}`. Task must save the scientific
submission below. Meta must preserve this interface when developing successor
Agents and arranging explicit reuse of scientific artifacts.

Task-specific execution instructions follow this common submission contract.
They are also available as `PROBLEM_DIR/execution.md`, with shared guidance
expanded only where requested by the task. Input shapes, target definitions and
native prediction formats remain in the task's README and data description.

## Research and final evaluation

For tasks without official validation, select_validation registers the exact
acquired sample IDs chosen by Task. Each call replaces the internal selection;
pass [] to clear it. Task may use its own splitting tools and revise membership
across generations. The host does not expand groups or date ranges or perform
validation preflight. Metric results and failures come from actual scoring.
Newly acquired labels are not automatically added to internal validation.
Refresh sample roles through the public reader after changing the selection.
Official V is retained; Test and unqueried labels remain unavailable.

Register internal validation before evaluation so the host knows which cohort
to score; official V needs no new declaration. Registration is not a prerequisite
for fitting. Accurately declare actual fitting membership and checkpoint ancestry.
The audit records overlap between fitting and the evaluated validation cohort.
Scores on overlapping samples are not independent estimates of generalization;
Meta must account for the recorded overlap when comparing evidence.

Bash uses `pipefail`, so `timeout ... | tail` reports the program's failure.
All child processes are stopped when a tool call returns, even after shell
success. Use foreground programs or `wait` within the same call. Background-job
cleanup returns an explicit error; do not spend later calls polling that job.

Save the complete frozen inference entry, checkpoints and required declarations
before calling `evaluate`. During research, `evaluate` snapshots the saved model,
inference entry and `frozen_model.json`, then executes that entry on a fresh,
label-free view of the currently authorized validation inputs. Only predictions reproduced by this
host run are scored; prewritten validation predictions do not qualify a model.
The exact artifact hashes and inference result are recorded with the score.

Meta self-evolution must not remove or disable final_selection_only or its
read-only history access and final artifact selection/submission interfaces.
Meta owns final scientific model selection. After research closes (at the
configured research time limit, the configured round limit, or an earlier research failure),
the current Meta receives a read-only final review with its own time limit
(default 300 seconds). No time is reserved from the research budget. API calls
and tokens remain in the cumulative ledger and obey the same configured limits;
review time and API usage are reported separately. User termination skips review
and Test.

Use `list_history`, `read_record`, `read_version` and `read_artifact` to inspect
existing trajectories, verified validation scores and cohort membership. Call
`select_final_artifact(artifact, reason)` to record a chosen existing artifact
and the comparison rationale, then finish the review. Alternatively,
`finish_experiment(artifact, reason)` selects and finishes in one call. These
selection tools are available only after research closes. The host allows only
history reads, model calls and selection in this phase: no shell/Python tools,
file edits, Agent updates, training, data fetching, label queries or evaluation.
No editable checkout or scientific data sandbox is created. If no valid
explicit Meta selection exists, final testing fails before Test access; the host
never picks a model based on highest score or silently falls back to another artifact.

Full recorded research conversations and command logs are available read-only
through `list_history_files(run_id)` and `read_history_file(run_id, path, offset,
limit)`. The file index includes completed API requests/responses (including
recorded failures), original prompts, tool results and unabridged command logs.
Only completed research is published; private scorer files, credentials and
Test records are excluded. Both Meta modes have the same access. During research,
Meta can also process this archive with its own code under `HISTORY_DIR`.

Access does not automatically put file contents into model prompts. Listings
contain metadata only (20 files by default); a read returns at most 4096 UTF-8
bytes by default, with `next_offset` for continuation and a configurable maximum
of 50000 bytes per read. `read_record` also returns paginated text by default
(character offsets; 4096 characters). Text pages also bound the JSON-escaped UTF-8
text to 24 KiB; a page may be shorter than `limit`. Always continue with the returned
`next_offset`, never by adding `limit`. The archive has no total-content truncation.

`list_history` returns a compact index (default 20 rows, also bounded by bytes),
with exact headline metrics, validation cohort identifiers and audit flags when
available. Large score tables and audit details remain in the exact record.
Follow `next_offset` and `next_version_offset` independently. Runs with many
evaluations or artifacts include `total_evaluations`/`total_artifacts` and
`next_evaluation_offset`/`next_artifact_offset`; use
`read_record(run_id, section="evaluations" or "artifacts", offset, limit)` for
entry pages. Omit `section` to read exact record JSON. Entries retain their original
order; the host does not rank or filter candidates for Meta.

Without `path`, `read_artifact` and `read_version` return metadata and a paginated
file manifest (default 20 files); `offset`/`limit` then count file entries.
Use `section="manifest"` to read the exact manifest JSON by character offset, or
`path` to read a file. Large tool responses have a 32 KiB display limit and an
explicit `_tool_result.call_id` for exact paginated recovery through
`read_tool_result`. Conversation archives retain exact messages; their bounded,
deduplicated reminders do not replace the original evidence.
Minimize reads: identify the current decision and missing evidence, inspect metadata
or concise notes, then read only relevant files and ranges. Avoid repeated or
duplicate content and stop once the evidence is sufficient. Do not replay complete
conversations or read all pages by default. Retrieval, indexing, summarization and
memory strategies belong to Meta. Reading
and repeatedly carrying content in model requests still consumes input tokens;
these interfaces do not guarantee a token saving. Final selection retains its
read-only restrictions and permits no new scripts or Agent changes.

The host checks the selected artifact's validation verification, declaration and
full file hashes, then runs its saved inference program once on Test inputs.
A changed artifact or failed inference causes final-test failure; there is no
retry or model selection based on Test. `meta_selection.json` records Meta's
decision; `final_selection.json` records its eligibility and validation evidence.
Each evaluation includes a `validation_cohort` sample count and membership hash.
Official V remains fixed. Internal validation contains the Agent's currently
selected acquired samples. Each evaluation retains its own membership and score.
When comparing rounds, account for changes in the model, training data and
validation samples. No historical rescoring is performed outside the research budget.
Final scores are not returned to Agents. After selection, no further Task/Meta
or their language-model API calls, annotation queries, training or method changes
are allowed during Test. The final Test time limit,
including input export, inference and scoring, is {{FINAL_SECONDS}} seconds.

Research actions and validation share the remaining research budget, with no
per-stage time limit or separate closing reserve. Save the fitted model and
inference entry early enough to evaluate them within `remaining_solve_seconds`.
Evaluation never extends the shared research budget. `host_scoring_seconds` in
feedback reports actual scorer duration so Task and Meta can plan subsequent work.

## Files to save in the scientific workspace

Save `frozen_model.json` at the workspace root, alongside the inference program
and its dependencies. For example:

```json
{
  "task": "{{TASK_ID}}",
  "method": "my-method-v1",
  "inference_entry": "predict.py",
  "training_entry": "train.py",
  "training_provenance": "training_provenance.json",
  "checkpoints": ["model.pt", "preprocess.json"]
}
```

- `task` must equal the current task ID; `method` must be a nonempty string.
- `inference_entry` names a saved Python file relative to the workspace; it
  defaults to `predict.py` if omitted. The host executes it as `__main__` with the
  final workspace as its working directory.
- `checkpoints` is a required, nonempty list of relative paths to existing saved
  model/parameter and fitted preprocessing files. It is not a prescribed directory
  name. Filenames above are examples; a non-neural method can save its fitted
  parameters in another format, such as `model.json`.
- Preserve the actual fitting program and name it with `training_entry`, including
  when training was launched with `python -m`. Declare dynamically loaded local
  modules in the optional `source_dependencies` list. Declared files must exist
  in the saved artifact. The host retains these files and execution records for
  provenance. There is no independent post-research LLM validity review.

Include all local inference dependencies and any previously acquired evidence
your method needs. Use paths relative to the workspace or the supplied environment
variables; the final workspace has a different absolute path. Keep required files
outside `output/`, `.tmp/`, and `__pycache__/`, which are omitted when the host
copies the artifact for final inference. Keep saved inference artifacts within
the workspace; symlinks are rejected.
During verified validation and final inference, the host makes the saved
workspace read-only, except for `OUTPUT_DIR` and `TMPDIR`. Write predictions to
`OUTPUT_DIR` and temporary files or caches to `TMPDIR`. The host creates a separate
`OUTPUT_DIR` for each run (validation and final) before the entry starts. The declaration,
checkpoints, programs and their saved dependencies cannot be changed during
these runs. Final inference must not fit preprocessing again on Test inputs.

Also save `training_provenance.json` (or the path named by `training_provenance`):

```json
{
  "labeled_fit_ids_file": "labeled_fit_ids.txt",
  "input_fit_ids_file": "input_fit_ids.txt",
  "parent_artifacts": []
}
```

Both UTF-8 ID files are required, with one sample ID per line; empty files are
allowed when no such fitting occurred. Stream large fitting ID lists to these
files instead of copying them into model context. List every label used to fit
model or preprocessing parameters in `labeled_fit_ids_file`, and every input used for
fitting (including unlabeled inputs) in `input_fit_ids_file`. List inherited
checkpoint artifact IDs in `parent_artifacts`; use an empty list for fresh
parameters. Preserve ancestors for resumed fitting, and include their fitting
membership. These files must live outside excluded output/cache directories.
The host checks the declared lineage against public research membership, rejects
Test or unknown members and unqueried label use, and binds the audit to artifact
hashes. It records validation fitting overlap, including inherited checkpoints.
The host also records executed scientific commands, bounded source snapshots
before and after them, and actual workspace inheritance independently of this
declaration. This evidence is available in the read-only research history;
missing or truncated observations are identified.
The membership audit checks the declaration. Neither source observations nor
read-only model files prove the actual data flow of arbitrary scientific Python
or exclude fitting in memory. Accurately record fitting and retain its code and
execution trace for reproducibility and later audits.

## Reading validation and final inputs

The host provides `PROBLEM_DIR/reader.py` and a fresh `DATA_DIR/batches.json`
listing batches in original `ordinal` order. During verified validation, these
are authorized Val inputs only; during final inference, they are Test inputs
only. Both views have hidden labels; it does not expose the research pool or research
batch paths. The environment sets `MEDRSEE_PHASE=final`, `MEDRSEE_SPLIT=T`, and
`MEDRSEE_TASK={{TASK_ID}}` for Test. For validation it sets
`MEDRSEE_PHASE=validation` and `MEDRSEE_SPLIT=V` (official V) or `validation`
(internal Val). Use `os.environ["MEDRSEE_SPLIT"]` in prediction metadata and
write directly to `OUTPUT_DIR` in both phases. Follow the reader guidance in
the task-specific execution instructions.

## Prediction outputs and handoff

The saved inference entry writes `OUTPUT_DIR/predictions.json` and
`OUTPUT_DIR/predictions/` directly in both phases. The host chooses `OUTPUT_DIR`
so replay outputs are separate from the original scientific workspace. Metadata
contains the current task ID, `split=os.environ["MEDRSEE_SPLIT"]`, and `count`
equal to the number of all supplied samples. Do not append a `validation/`
subdirectory in the inference entry. Follow the task's README and data description
for exact filenames, arrays, shapes, classes and geometry. Preserve ascending
`ordinal` order across batches.
The host copies the declaration into the prediction output before scoring.

Save a complete reusable submission before each Task handoff and before the shared
research budget expires. `evaluate` snapshots the workspace; handoff also saves
pending workspace changes. A successful verified validation score establishes that the saved entry ran
on those validation inputs; it does not guarantee success on every Test input. Meta can arrange reuse with Task `artifact_parent`; artifacts are not
inherited automatically.
