# execute_place

Execute the current placement plan with the bound held object, open the gripper, detach and execute the verified open-hand retreat. Preserve actual feedback and partial-effect evidence on error. It neither chooses a new landing point nor marks the target delivered. verify_place separately checks the outcome.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
