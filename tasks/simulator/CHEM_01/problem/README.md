# CHEM_01 — ChemGymRL water/oil extraction

Choose operations from the native WaterOilExtract-v0 simulator to maximize the cumulative
reward. The unchanged native terminal reward compares the final and initial values of
`sum(recoverable_NaCl_moles - C6H14_moles)` over the two working vessels (the extraction
vessel and collection beaker). Recoverable NaCl includes the native dissolved-component
contribution; the waste vessel is outside this sum. Thus retained hexane reduces the
objective, and both salt retention and oil removal matter.

The cumulative episode return adds native operation penalties to that terminal difference.
Each vessel-operation feedback code -1 incurs a -0.1 penalty. This native feedback penalty
is distinct from API argument validation errors; there is no universal fixed penalty on
every step. Rewards are not a purity ratio. Immediate termination leaves the physical
objective unchanged and gives 0.
Use the current observable vessel state and feedback; full simulator state is organizer-only.

`simulator_action` uses action=`step`, arguments={"action": n} with integer n in 0..40.
Read DATA_DIR/action_schema.json for operation/parameter mappings; action 40 terminates.
The native horizon is 50 steps. The raw primary metric is `episode_return`, higher is better.
Only a terminated episode has a valid score.

The initial physical/RNG state is the existing fixed seed 20260921 snapshot.

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
