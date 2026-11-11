# Adapted from facebookresearch/HyperAgents, commit 59a68f672dfb92c74aeb7e61535d776fb36e172d.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# CC BY-NC-SA 4.0; see HYPERAGENTS-LICENSE.md in the repository root.
def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* Use the installed Python environment; package installation is unavailable.
* Files persist. Each command starts a new shell in WORKSPACE; shell variables and background processes do not persist. All remaining children are stopped even after a successful shell exit. Run foreground commands or wait within this call; do not use nohup and poll later. Bash uses pipefail so a failure before tail/tee remains visible.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Run commands in the foreground. Execution is capped at 120 seconds and the shared remaining_solve_seconds; each invocation counts as one tool call.""",
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
