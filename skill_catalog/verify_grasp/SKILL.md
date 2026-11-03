# verify_grasp

Check the actual gripper and lift/holding evidence for the bound target and arm. Return success or an explicit failed/unknown condition. It does not move, recapture, plan, silently close again or install an attachment before lift.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
