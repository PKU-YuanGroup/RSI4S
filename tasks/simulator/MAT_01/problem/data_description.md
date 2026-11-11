# Data and feedback

DATA_DIR contains known_structure.cif, schema.json and task_index.json.
In schema.json, each actions entry has an arguments_schema describing the exact
simulator_action arguments and an example containing the complete action/arguments object.
task_index.json provides the public initial motor positions, absolute motor ranges and ROIs.
`simulator_observe` returns task, observation, terminal, metrics, revision and usage.
Initial observation instructs you to obtain wh/get_sample_information. After an action,
observation is the original response, including detector images as JSON arrays and scan
results when applicable. usage tracks commands/max_commands and exposures/max_exposures.
Motor status and the known crystal structure are public; true orientation, hidden offsets
and random state are organizer-only. Do not read hidden states or reference fixtures.

There are no train/validation/test members. The single demo uses the fixed original
Co3Sn2S2 configuration; the score is an orientation angle in degrees.

Source: llm4xray-v0.0.0 / existing ScienceGym MAT_01 frozen recipe. The original
source and license remain in the prepared external data directory.
