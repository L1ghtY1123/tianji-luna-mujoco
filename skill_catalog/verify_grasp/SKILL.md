# verify_grasp

Check bound grasp execution and current mechanical gripper feedback for the execution-layer carrying model. No Decisions call or new image capture. Do not infer object identity, location, processing state or completion from the gripper, and do not write held/not-held state to instance memory. This skill does not move, plan or close again.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
