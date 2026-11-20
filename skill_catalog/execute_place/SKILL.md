# execute_place

Execute approach, place and retreat under one position-mode Base enable scope. Keep the closed command through transport. At the release waypoint validate measured TCP, the opening sweep and the retreat connector before sending the open command. Wait for measured gripper completion. The optional post_gripper_hold_s defaults to zero. Validate carried motion, opening and detached retreat before execution. Record actual release TCP and action receipt, detach carrying geometry, and invalidate the affected spatial layer. Graph calls verify_place to judge the final destination from current RGBD.

Use the shared cuRoboV2-Fast v0.9.0 service and its official solver defaults. Inputs, prerequisites, effects, capture permissions and outcomes are defined in `contract.json`.
