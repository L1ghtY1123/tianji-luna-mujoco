# choose_arm

Choose among arms permitted by current geometry, measured robot state and offline workspace. Use the sole admissible arm directly; when several remain, Decisions proposes one. This proposal is not proof of grasp IK or trajectory feasibility. Known CAD grasps must come from the selected arm offline library. Do not impose a later processing capability arm on an earlier transfer.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
