# return_to_initial_pose

Return both empty hands to the actual joint posture recorded at task start, using the current or explicitly authorized retained collision scene. If already there, verify measured joints without movement. This is distinct from the operator observation posture. No implicit observation or completion judgment is inserted. The graph includes this skill when the user requires final return.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
