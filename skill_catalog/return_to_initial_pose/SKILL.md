# return_to_initial_pose

An unconfirmed destination relationship does not itself authorize or forbid motion. After release, returning requires the original completed release/open-gripper feedback, a current observation collision map containing the target measured after release. Graph-selected sensing/mapping must establish those prerequisites first. This case cannot use the retained pre-release map, and return does not confirm placement.

Return both empty hands to the actual joint posture recorded at task start, using the current or explicitly authorized retained collision scene. If already there, verify measured joints without movement. This is distinct from the operator observation posture. No implicit observation or completion judgment is inserted. The graph includes this skill when the user requires final return.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
