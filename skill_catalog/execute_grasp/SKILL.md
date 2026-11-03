# execute_grasp

Execute the selected current approach/grasp trajectory, close the gripper, validate actual closure/start/collision state, then execute the planned lift. The first lift has no object attachment, as configured by the user. Preserve partial physical effects on failure; never replay them as an ordinary retry. Return measured execution/holding evidence. Attachment is prepared only after successful lift and holding confirmation.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
