# move_to_observe

Plan and execute an explicit empty-hand dual-arm camera movement against a current observation map or an authorized retained scene. viewpoint=configured returns to the operator observation posture. wrist_tilt_positive/negative offset both sixth joints by +12/-12 degrees to propose distinct wrist-camera views; these require joint-limit, collision and execution checks and do not certify visibility. Alternative viewpoints do not satisfy a required return to the configured posture. Capture at the reached viewpoint, then return for the graph to choose analysis. Do not repeat an unchanged alternative viewpoint. The skill does not insert observe_scene, verify_place or a retry. If no current/authorized collision scene exists, motion is unavailable; a branch cannot waive that requirement.

Recovery viewpoints require a current Decisions diagnosis of not_visible, cropped or occluded evidence. CAD mismatch, visible identity ambiguity, API errors and planning failures alone cannot authorize movement. A normal configured return required by the task remains a separate motion purpose. The diagnostic and recovery selection share one request; numerical movement checks still apply.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
Recovery may explicitly retry this skill after repairing a rejected prerequisite. Each invocation plans a new collision-checked trajectory from current measured joints; it never replays the old path. The runtime still blocks unresolved partial motor/effect execution. A normal observation map uses `build_scene.exclusion_state=observation`; `selected` is not that map. An authorized retained scene remains an existing alternative. No repair or retry is inserted automatically, and changing viewpoint for visibility still requires a current visual-limitation diagnosis.

# 地图前提

`observation_motion_scene_ready` 是现有数值入口检查的事实：地图未失效，且为与当前观测绑定的 observation 地图，或已有允许回位的保留场景。selected 地图本身不满足这一要求。该事实只表达能否调用，不能绕过实际规划、碰撞和执行验证。

