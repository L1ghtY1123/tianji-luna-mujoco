# move_to_observe

For a released object whose destination semantics remain unknown, require the original completed release/open-gripper feedback, a current observation collision map containing that target measured after release. Establish this map with graph-selected observe_scene/build_scene before moving. Do not use the retained pre-release map. Moving does not confirm placement or change the pending transfer's verification status; resume verify_place with the original release record.

Plan and execute an explicit empty-hand dual-arm camera movement against a current observation map or an authorized retained scene. viewpoint=configured returns to the operator observation posture. wrist_tilt_positive/negative offset both sixth joints by +12/-12 degrees to propose distinct wrist-camera views; these require joint-limit, collision and execution checks and do not certify visibility. Alternative viewpoints do not satisfy a required return to the configured posture. Capture at the reached viewpoint, then return for the graph to choose analysis. Do not repeat an unchanged alternative viewpoint. The skill does not insert observe_scene, verify_place or a retry. If no current/authorized collision scene exists, motion is unavailable; a branch cannot waive that requirement.

For recovery, prefer the configured observation posture, followed by an explicitly authored observe_scene and restoration of invalidated prerequisites. Decisions selects this existing branch. A configured return does not require another judgment proving occlusion. Empty hands, an authorized observation-motion map, joint limits, collision checks and execution feedback remain mandatory. Reobservation is an evidence-repair attempt, not proof that a numerical planning failure has been solved.

Alternative wrist viewpoints require one current Decisions judgment that task-relevant visual evidence is unavailable because of being outside the images, cropped or occluded. Judge this common limitation directly for the original failed skill and bound subject ID; the exact cause may remain uncertain. Do not sum cause-specific probabilities or duplicate this question in a preceding router.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
Recovery may explicitly retry this skill after repairing a rejected prerequisite. Each invocation plans a new collision-checked trajectory from current measured joints; it never replays the old path. The runtime still blocks unresolved partial motor/effect execution. A normal observation map uses `build_scene.exclusion_state=observation`; `selected` is not that map. An authorized retained scene remains an existing alternative. No repair or retry is inserted automatically. Repeating an unchanged capture reuses its answer; finite recovery cannot retry the same evidence until yes.

# 地图前提

`observation_motion_scene_ready` 是现有数值入口检查的事实：地图未失效，且为与当前观测绑定的 observation 地图，或已有允许回位的保留场景。selected 地图本身不满足这一要求。该事实只表达能否调用，不能绕过实际规划、碰撞和执行验证。

Successful intermediate mapping preserves the failed subject. Successful retry of that operation clears its active failure, while the event history is retained.
