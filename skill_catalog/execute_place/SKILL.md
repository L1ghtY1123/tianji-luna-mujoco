# execute_place

Execute the bound approach, place and retreat trajectories on one time axis. Open the gripper after the place phase using gripper_transition_s. Validate carried motion, opening and detached retreat before execution. Record actual release TCP and action receipt, detach carrying geometry, and invalidate the affected spatial layer. Graph calls verify_place to judge the final destination from current RGBD.

Use the shared cuRoboV2-Fast v0.9.0 service and its official solver defaults. Inputs, prerequisites, effects, capture permissions and outcomes are defined in `contract.json`.
