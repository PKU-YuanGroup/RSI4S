# Data and feedback

DATA_DIR contains action_schema.json, initial_observation.json and schema.json.
`simulator_observe` returns task, observation, terminal, metrics, revision and usage.
Initially observation is [native_observation_array, info]. After a step it is
[native_observation_array, reward, terminated, truncated, info]. All arrays are JSON lists.
usage tracks steps and max_steps=50. metrics is absent/null until native termination.

There are no static labeled examples or train/validation/test members. Public snapshots
are reference observations; the host-owned live episode is authoritative. Do not use
hidden simulator state, initial-state snapshots or organizer dynamics fixtures.

Source: ChemGymRL commit ab8227b6b33f13617b7e551bdf6b894df7eec68d, existing
ScienceGym CHEM_01 frozen recipe. Third-party code remains external under its original license.
