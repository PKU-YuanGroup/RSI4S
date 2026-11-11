# Adapted from facebookresearch/HyperAgents, commit 59a68f672dfb92c74aeb7e61535d776fb36e172d.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# CC BY-NC-SA 4.0; see HYPERAGENTS-LICENSE.md in the repository root.
def tool_info():
    return {
        "name": "bash",
        "description": """Run bash in WORKSPACE with pipefail: a failed or timed-out program stays a failure even when piped to tail/tee. Files persist; each call starts a new shell. No network or installs. Run programs in the foreground, or wait for all children within this call. ALL remaining child processes are stopped whenever the call ends, including a successful shell exit; nohup, '&', and daemonization do not keep jobs alive. Never sleep/poll in later calls for such jobs. Foreground execution: at most 3600 seconds, capped by remaining stage/shared time. Counts as one tool call. Save usable outputs before long runs and leave time for later Agents. Read DATA_DIR/PROBLEM_DIR; save and execute run.py; write outputs to OUTPUT_DIR. Fit and infer on assigned CUDA; CPU preparation is allowed.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }
