# identify_destination

Bind an observed support/container or an operator-configured work area for the current transfer. Preserve the original location requirement. A designated/set/configured area means the workbench setting and uses its exact XY; do not search for or visually invent a marker. A normal table/container permits region-based placement. Intermediate destinations need explicit semantic permission. This does not choose a collision-free XYZ or plan a path.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
中转许可：Decisions 明确否定时返回 failure/intermediate_destination_rejected，只有证据不确定才返回 unknown。本能力不改写任务的最终目的地；输入物体 subgraph 的 destination_ref 覆盖加工谱系的最终交付，work_area_ref 表示加工/暂存区域。
