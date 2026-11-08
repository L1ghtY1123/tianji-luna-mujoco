# execute_place

Execute only the phases of the bound release_mode. surface_setdown executes approach/place, opens and detaches, then executes the checked retreat. container_release executes the single carried place trajectory, stops, verifies actual complete-object bottom and opening footprint, opens and detaches; no automatic retreat or observation follows. Preserve release receipts and invalidate changed spatial geometry before any subsequent planning. verify_place judges the landing separately.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

该动作清除 observation_pose_ready，后续要求观测位姿时由图显式安排回位。

## 物理回执与时间记忆

实际夹爪开位确认后、撤离前持久化开爪事件。撤离失败不丢失开爪记录；张开本身不证明物体离手，不覆盖已记录的视觉支撑和位置，也不写入物体 released。物体支撑和目的区域由视觉核验分别判定，不询问夹持/未夹持。

The plan carries MotionPlanBinding for target, destination, arm, generation, observation, scene and settings. Execution validates this binding before physical commands. Grasp planning consumes a detached scene/observation snapshot throughout its bounded search.
