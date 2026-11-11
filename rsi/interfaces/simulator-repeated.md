Task {{TASK_ID}} uses the repeated_fixed_episode_v1 research protocol: each Task stage works on
one episode of the same fixed scenario, then Meta improves the next Task, which starts a fresh episode.
The scientific simulator, starting physical/RNG state, allowed actions and score are unchanged.

The campaign has one cumulative research clock and API ledger across all Task/Meta generations.
The host starts a fresh copy of the SAME fixed scenario before the next Task ONLY after the previous
episode is terminal. An unfinished episode continues with its spent acquisition budget intact.
Agents cannot reset, reseed or create episodes; Meta can observe and develop Task but cannot act.
Each new episode has the full acquisition allowance (ChemGym 50 steps; X-ray 128 commands and
64 exposures). Starting a new episode does not reset the campaign clock, API cost or score history.
Simulator crashes and ambiguous transport failures cannot be erased by requesting a new episode.

This is repeated optimization on one development scenario. There is no independent Test,
train/validation split or D0/Q pool. Do not describe repeated scores as held-out generalization.
The campaign result is the best completed episode with a valid primary score; every episode
remains in the score history.
Use the cumulative history to develop and improve the policy based on actual measurements/rewards.

Use simulator_observe for the current permitted observation, episode_index, protocol and usage.
Only Task may use simulator_action or simulator_run. Read the public task schema for exact argument keys.
X-ray must form an orientation with or0/or1 before finish; finish with no orientation is a failed
episode, not a zero-error solution. ChemGym action 40 or the 50-step horizon terminates its episode.

For simulator_run, write a workspace Python file defining next_action(observation, state).
observation is the full simulator_observe object; state starts as None per simulator_run call.
Return {"action":"...","arguments":{...},"state":...}, or {"stop":true,"state":...}
to yield without ending the environment. Each decision runs in a fresh confined Python process.
Use JSON state or workspace files for persistence. The simulator remains in its trusted process.
Policy errors/timeouts retain consumed actions. Observe the current state before resuming.

Save the policy, fitting/optimization code and notes BEFORE the terminal simulator action.
On terminal completion the host automatically snapshots, scores and ends the Task stage, then
allows Meta to publish the next Task. Further calls on a closed episode are not useful.
evaluate on an unfinished episode returns episode_pending; it cannot create an independent score.
The trusted simulator supplies scores, tied to episode indices, trajectories and Agent versions.
Output files cannot provide or override metrics. No predictions.json or training declaration is required.

Meta uses measured scores and full public trajectories to improve the successor Task. Preserve
successful policy artifacts explicitly when useful. Repeated evaluation without a reasoned strategy
change does not constitute improvement. Terminating one episode ends that Task stage, not the campaign.
The global time budget and configured generation limit still apply. Do not read hidden state or fixtures.
