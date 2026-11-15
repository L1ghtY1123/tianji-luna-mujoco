# Scene reference binding implementation plan

The user requires correct blue-bin selection and prevention of pink-object/blue-bin identity confusion. Optional scene descriptions supplement the natural-language task.

## Existing interfaces

`ObjectRegistrar.bind_proposals` consumes Responses proposals and current SAM3 detections. `current_instance_groups` establishes current camera groups. `choose_identity` compares dated instances. `bind_compiled_task` binds scene references to accepted permanent IDs. Database transactions and model evidence remain the persistence boundaries.

## Implementation and verification

- [x] Replace instance-to-description queries in `agent/registration.py` with description-to-current-group choice queries. Present all current groups together in the same three RGBs. Keep unique group ownership and unresolved reference outcomes.
- [x] Confirm unselected physical groups independently and register them through the existing path, preserving new-instance workspace requirements and historical identity comparison.
- [x] Update `agent/decision_state.py` and current contracts to describe the new question direction. Remove the replaced matcher and migrate its useful checks. The existing graph prompt already requests all visible objects and retains its current interface.
- [x] Reproduce archived temporal cross-category and left/right-bin failures with the current payload format. Shuffle candidate order to check label grounding. Inspect each failed sample before changing temporal behavior.
- [x] Exercise real `bind_proposals` with frozen RGBD/SAM3 and an isolated database, including the missing-right-bin description case. Verify view membership, distinct IDs and task binding. Existing transaction rollback/retry checks pass.
- [x] Run local regression, compile and frontend checks as affected. Deploy backend changes with exact before/after hashes while idle. Replay labels are under final verification.
- [x] Run the user-authorized pink-object transfer with a scene description through normal Responses and graph execution. Inspect actual identity decisions and resulting object records. Preserve every failed attempt and report remaining limits.

## Acceptance

Both blue bins remain separate, the left-bin task reference selects the correct group, and the pink object's ID stays attached to its current physical masks. A successful transfer requires the existing grasp, placement and task-completion evidence. Responses remains one call per task. Existing planning, collision, timing and post-Lift attachment requirements apply.

## Grasp and console follow-up

- [x] Reproduce the false-empty holding judgment with archived images and compare single-wrist questions. Validate the selected production input on positive and negative captures.
- [x] Replay the failed motion in isolated MuJoCo, inspect held-object motion, and compare contact, linkage and arm-servo settings. Verify the deployed model through repeated loaded transport.
- [x] Verify empty terminal return and gripper opening through the production terminal graph in an isolated simulation.
- [x] Hide immutable settings in the editor and verify editable values and preserved backend settings in a browser.
- [x] Complete the normal transfer and verify archived Decisions images, identity records and actual placement.

Latest normal run `039d70dbe1a94198805a7df0bafe1751` succeeded in 185.226 s with one Responses call. Pink ID 769, left bin 767 and right bin 768 remained distinct and consistent. Actual object footprint is inside the configured destination. Replay verification checked 31 Decisions calls against archived image paths. Loaded transport still has approximately 13.8 mm total relative translation after additional round trips, so the material/servo values remain simulator calibration assumptions.
