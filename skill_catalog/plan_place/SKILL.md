# plan_place

Plan the current resolved placement using the official grasp-style approach and place phases with the post-lift attachment, then an open-gripper retreat after release. Preserve the selected XYZ/orientation and current world binding. Validate the phases with the configured hard collision clearance. No additional pose-selection strategy or cross-skill transport is inserted. It does not release or declare visual success.

The resolved single goal uses the official BatchMotionPlanner with batch capacity 1 for the approach/place phases. Batch capacity does not create additional placement candidates; retreat and trajectory validation remain required.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
