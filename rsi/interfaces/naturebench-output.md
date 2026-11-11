# Platform submission interface: NatureBench

This interface applies to task `{{TASK_ID}}`. Task must save its scientific code
and outputs in the workspace. Follow the task's README and data description for
its end-to-end entry point (normally `run.py`), instance directories, prediction
filenames, formats and scientific constraints. Write scored outputs under
`OUTPUT_DIR`; `evaluate` snapshots the workspace and invokes the existing
task-specific evaluator. A final response ends the Task stage; it does not replace
the required output files.

The current NatureBench adapter scores task-specific outputs during research. It
does not add a separate frozen-inference final Test stage. It does not require
`frozen_model.json`, an `inference_entry` field, or a `checkpoints` list as a
universal submission format. Any model files explicitly required by an individual
task must still be saved. No independent post-research LLM validity review runs.

Meta must preserve the task's submission requirements when developing successor
Agents. Save reusable code, fitted artifacts and exact evaluation results before
handoff or budget exhaustion. Meta can arrange scientific-artifact reuse with
Task `artifact_parent`; artifacts are not inherited automatically.
