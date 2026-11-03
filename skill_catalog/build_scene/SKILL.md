# build_scene

For a local update, entity_ids selects CAD/OBB object layers. Declared table IDs may accompany the object IDs: their measured cuboids use the separate support input and remain included. A nonempty local selection needs at least one object ID; table-only selections, marker IDs and configured region names are not object layers. Global uses an empty entity_ids list. The table permanent ID remains bound to its cuboid.

Publish the selected global/local inventory collision layers: accepted current CAD meshes, measured unknown-object OBBs and table cuboids. Reuse compatible resident GPU planners. This does not capture images, run SAM3 or FoundationPose, or integrate TSDF/ESDF in CAD/OBB mode. Global updates all currently observed inventory entities; local updates only the selected entities and preserves valid other layers. Neither means hidden space has been observed. The graph chooses scope and exclusion_state. observation keeps all obstacles for empty-hand viewpoint motion; selected binds the current grasp target. A selected map is not an observation map. Missing or stale required geometry is reported, never repaired by republishing identical data.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
The argument-dependent effects include `observation_map_ready`: true for a successfully published complete `observation` map, false for `selected`. Publishing a selected-target map does not make it an observation-motion map. This distinction is provided to Responses/Decisions; the graph still chooses the argument and subsequent action.

# 回位地图效果

成功发布 observation 地图会提供 `observation_motion_scene_ready`；selected 模式不保证这一事实。运行时根据地图版本及已验证转移状态计算实际值，恢复选择可以看到原失败节点缺少的准确前提。

