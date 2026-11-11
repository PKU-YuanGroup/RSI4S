# MAT_01 — AI X-ray scientist crystal orientation

Operate the offline six-circle diffractometer for the original Co3Sn2S2 demo crystal.
Use the known structure and measured diffraction feedback to estimate its orientation.
The simulator has hidden motor offsets and detector randomness; their original initial state
is frozen. No physical instrument is connected.

Allowed simulator_action names: wh, get_sample_information, umv, pd_exposure, dscan, or0,
or1, finish. See DATA_DIR/schema.json for arguments and ranges. The budget is 128 commands
and 64 exposures, including each scan point. Failed simulator operations can consume budget.
finish takes no arguments and remains allowed after the command budget.

## Exact action arguments

Pass `{"action": "command_name", "arguments": {...}}` to `simulator_action`.
`schema.json` gives an `arguments_schema` and a directly usable `example` for every action.
Use the following exact keys inside `arguments`; names are case-sensitive:

| Action | Arguments |
| --- | --- |
| `wh`, `get_sample_information`, `finish` | `{}` |
| `umv` | One or more motor names directly, e.g. `{"delta": 20.0, "eta": 10.0}`; absolute angles in degrees, within `task_index.json` motor ranges. |
| `pd_exposure` | `{"time": 1.0}`; time in seconds, between 0.001 and 10. |
| `dscan` | `motor`, `start`, `finish`, `intervals`, `time`, `roi_name`; all six keys required. |
| `or0`, `or1` | Separate numeric `H`, `K`, `L` keys, e.g. `{"H": 0, "K": 0, "L": 6}`; each between -100 and 100. |

For example, this scans delta from 0.1 degrees below its current position to 0.1 degrees
above, acquiring two exposures of one second each:

```json
{"action": "dscan", "arguments": {"motor": "delta", "start": -0.1, "finish": 0.1, "intervals": 2, "time": 1.0, "roi_name": "pd6"}}
```

`start` and `finish` are relative offsets; both resulting absolute endpoints must stay
within the motor range. `intervals` is the number of exposure points (2–32), including
endpoints. ROI names are `pd5`, `pd6`, `pd7`. The `finish` argument inside `dscan` is
the scan's endpoint offset; the separate `finish` action ends the experiment.
Pass motor keys directly, without a `motors` wrapper. The adapter accepts `time`,
separate `start`/`finish`, and separate `H`/`K`/`L`; descriptive labels from the old
schema (`time_seconds`, `relative_start_finish`, `H_K_L`) are not argument names.

Examples illustrate API syntax. Choose actual positions and indexed reflections from
measurement feedback. `or0`/`or1` record the current geometry without taking an exposure;
invalid reference geometry can return `observation.result.success=false` even when the
argument format is accepted. Read that message as well as any top-level `error`.

The primary metric is symmetry-aware `relative_angle_degrees`, lower is better. Missing
orientation is an explicit failure. The original acos extrapolation is retained, including
the slightly nonzero angle for an identical orientation.

## Framework interface

Use `simulator_observe`, `simulator_action`, or `simulator_run` from Task. Meta can read observations
and history and evolve Task, but cannot perform actions. For programmatic control, write
`next_action(observation, state)` in a workspace Python file and call `simulator_run` with its path.
Return `{"action": "...", "arguments": {...}, "state": ...}`; return `{"stop": true}`
to hand over without terminating the environment. Each policy decision runs in a separate
confined process. The host stores the simulator outside the Agent sandbox.

Each Task stage works on one episode of this fixed scenario. A terminal episode ends that Task
stage, not the experiment: Meta reviews the result and improves the next Task, which receives a
fresh copy of the same initial scenario with the full action budget. An unfinished episode
(for example after `{"stop": true}`) continues in the next Task with its spent budget, including
failed programs. Agents cannot reset or reseed. There is no training, validation or independent
test set. `evaluate` reads the trusted terminal result; before completion it returns
`episode_pending`. The campaign clock, API ledger and score history are cumulative; the result is
the best completed episode with a valid primary score. These repeated development episodes do not
create an independent test set.
