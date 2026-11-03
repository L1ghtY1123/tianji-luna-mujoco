# Remove fixed unpack lifecycle from graph execution

User authorization: remove the old lifecycle and fixed workflow entirely. Keep Responses + Decisions + extensible skills, permanent entity IDs, observed inventory mapping, and the declared simulation unpack capability. No additional design approval is needed for this explicitly requested continuation.

## Shared interfaces

- Responses discovery returns `entities`, `reference`, `evidence`, `task_spec`, `policy_graph`, `task_inventory`. It does not create lifecycle rows or select a workflow. `task_inventory` holds `source_object_ids` (candidates), `selected_ids`, `source_id`, `destination_id`, `work_area_id`; these are bindings, not execution stages.
- The controller owns `task_inventory` and `policy_graph`. Observation reference contains `task_spec`, `policy_graph`, and `inventory_state` with `entities`, `retired_entity_ids`, `transformations`. No `demo_task`, `item_lifecycle`, phase chain, hardcoded two-leg route, or preselected future arm is required.
- `PolicyTaskProgress` extends the measured inventory utilities. `configure_task(task_spec, selected_ids)` binds goals. `register_product(product)` consumes an input only after verified transformation evidence, adds a separate product ID, and records lineage. `completed_input_ids()` checks verified final location plus requested processing effects. `confirm_visual_transfer` records each verified location without imposing a route. Intermediate release is not task completion. Repeating a skill after a failed partial action remains blocked.
- Physical skills use `self.state['target_id']`, `destination_id`, `arm`, context holding/action receipts, and task ledger records. Processing is an explicit capability, not a task workflow. `prepare_unpack`, `vla_unpack`, `verify_unpack` retain their skill-specific evidence requirements; no old phase or future-leg arm fields. Transformation receipts live in `context['processing_action']`; `verify_unpack` calls `task_progress.register_product(product)`, `self.sync_inventory(context)`, and returns `product_id` plus `output_entity_id`. Product dictionary includes `id`, `input_entity_id`, `process_id`, `processing_skill='vla_unpack'`, descriptions, segmentation query, and creation evidence. `sync_inventory` updates state/reference from ledger, not a lifecycle state machine.
- Release skills record `context['post_place_capture']`, `post_place_verification_input`, `place_verification`, and `pending_transfer`. VLA checks the verified current release and measured configured work area. Scene refresh and observation movement remain explicit graph skills. Native planning, collision and coverage checks remain intact.
- Configured unmarked XY regions are declared intent. Deterministic code binds them to measured support. Decisions confirms the visible supporting entity and object identity, not whether the supplied coordinates are visually identifiable. Fixed arm workspaces are numerical inputs; feasible-arm preference can use Decisions, reachability cannot.

## Work and verification

1. Remove lifecycle use from perception and source membership; keep current identity/geometry validation.
2. Replace unpack workflow orchestration with independent processing and container-transfer skills; remove obsolete lifecycle modules after moving reusable evidence utilities.
3. Replace compiler compatibility output and runtime two-leg accounting with generic bindings, transfer evidence and transformation lineage.
4. Update contracts, tests, UI state consumers and docs. Remove obsolete tests of intentionally deleted fixed stages; preserve or migrate physical/evidence regression coverage.
5. Verify no active old lifecycle dependencies, run focused and full suites, inspect actual model request data, deploy only this user's mounted service and verify real API behavior. Report physical acceptance limits explicitly.
