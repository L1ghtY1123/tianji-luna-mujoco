# build_scene

The initial collision publication must be global. `scope=local` requires `collision_map_initialized=true`, established by a successful complete global publication. Geometry invalidation does not erase this base; local refreshes may repair it. A new task or reset starts without a base. With `scope=decide` and no base, global is the sole permitted scope and needs no model judgment; after initialization Decisions may select global or local. An explicit local request is never silently changed to global.

`failure_preserves` declares that a failed or uncertain publication leaves `collision_map_initialized` unchanged. It neither creates a missing base nor deletes an existing one. This says nothing about current collision freshness or planning readiness.

For a local update, entity_ids selects CAD/OBB object layers. An empty local list uses the currently selected target_id, including a selected product. Explicit lists refresh exactly their selected object layers. Declared table IDs may accompany the object IDs. Their measured cuboids use the separate support input and remain included. A nonempty local selection needs at least one object ID. Table-only selections, marker IDs and configured region names are rejected as object selections. Global uses an empty entity_ids list. After the global base is initialized, scope=decide requests a separate mapping strategy judgment. The table permanent ID remains bound to its cuboid.

Publish the selected global/local inventory collision layers: accepted current CAD meshes, measured unknown-object OBBs and table cuboids. Reuse compatible resident GPU planners. This does not capture images, run SAM3 or FoundationPose, or integrate TSDF/ESDF in CAD/OBB mode. Global updates all currently observed inventory entities; local updates only the selected entities and preserves valid other layers. Neither means hidden space has been observed. The graph chooses scope and exclusion_state. observation keeps all obstacles for empty-hand viewpoint motion; selected binds the current grasp target. A selected map is not an observation map. Missing or stale required geometry is reported, never repaired by republishing identical data.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
The argument-dependent effects include `observation_map_ready`: true for a successfully published complete `observation` map, false for `selected`. Publishing a selected-target map does not make it an observation-motion map. This distinction is provided to Responses/Decisions; the graph still chooses the argument and subsequent action.

# 回位地图效果

成功发布 observation 地图会提供 `observation_motion_scene_ready`；selected 模式不保证这一事实。运行时根据地图版本及已验证转移状态计算实际值，恢复选择可以看到原失败节点缺少的准确前提。


The archived collision snapshot records the build node, chosen scope/reason, local entity IDs and excluded target. Each layer retains its actual enabled flag and geometry observation ID. A local update can publish a world that also contains retained layers; the archive does not claim all visible geometry was measured again.

碰撞依赖来自当前测量、保留图层、动作失效的物体以及当前目标和目的地。单纯的未见历史清单条目不新增必需图层。当前操作或受影响物体缺少有效几何时，继续返回明确失败。
