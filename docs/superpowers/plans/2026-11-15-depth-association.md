# Depth-assisted instance association implementation plan

> Execute inline using executing-plans; repository rules prohibit subagents and worktrees.

**Goal:** Prevent spatially separated same-capture masks from being merged, and supply action-aware historical spatial evidence to Luna identity questions.

**Architecture:** Reuse calibrated per-mask world points and dated RGBD/masks already in ObjectDatabase. A pure evidence module supplies measured summaries, conservative spatial comparisons and dated motion context. Decisions continues choosing identity from RGB plus this evidence; numeric contradiction checks cannot fabricate a model answer. No camera-to-camera silhouette projection, additional sensor capture, or new verification questions.

**Spec:** User-approved design in this chat: cross-camera RGB plus metric evidence; temporal evidence respects executed movements; verify_grasp unchanged. Subsequent user instruction unifies table/container verify_place and defers completed container contents for both success grades. Same-arm, fixed CAD and destinations; gpt-6-luna only.

**Alternatives:** Raw depth images add image cost and require model interpretation of scale. Fully geometric matching cannot establish semantic identity. Use existing RGB with compact world-frame depth measurements and a conservative same-capture contradiction check.

## Tasks

- [x] Reproduce the archived wrong-box choice through the existing correspondence entry point. Keep diagnostics in ignored data/runs/depth-association-20261115, not a new tests directory.
- [x] Add association_depth.py: candidate_depth_evidence(capture, candidate), spatial_relation(left, right), association_depth_state(capture, candidates, entities, reference). Keep per-camera candidates independent; invalid/insufficient depth remains unavailable.
- [x] Wire current_instances.py, task_subjects.py, temporal_identity.py and subject_assignment.py. Preserve model answers and separately record accepted choice and spatial rejection. Check transitive groups before merging. Historical positions are priors, never hard exclusion after motion.
- [x] Update observation cache dependencies for physical receipts, decision evidence version, current contracts and documentation. Reuse database RGBD history rather than duplicating sensor records or changing storage schema.
- [x] Check displaced faces, separated boxes, insufficient depth, moved camera calibration, executed release versus historical position, and observation cache invalidation using a local diagnostic script.
- [x] Replay saved failed box evidence with actual Luna; then run bounded current-API simulation experiments, inspect all results, synchronize source to the authorized Ubuntu deployment, and verify hashes/runtime version.

## Validation

Use .venv/Scripts/python.exe for the local diagnostic, ruff check src --select F, compileall, and the existing web build. Remote experiments are serial with unchanged service CPU limits. Archive source hashes, model inputs/answers, accepted associations and measured outcome. No SDK changes, no git commit/push unless requested again.

Ruling: Use one visible bounds diagonal plus 2 cm with a 15 cm floor for gross-separation checks. The initial two-diagonal allowance failed to reject the actual archived opposite boxes (81 cm gap); the final rule rejects them while keeping all three same-box controls.

Ruling: Reuse original SAM3 query/alias retrieval membership for cross-camera choices. v6 full simulation exposed box/table/workpiece confusion among unrelated query candidates. This extends the existing subject-selection retrieval rule, preserves fallback aliases and image context, and leaves same-query identity to RGB plus depth. Do not classify identity from SAM3 query membership.

v6 validation: 6/6 frozen box comparisons correct with real Luna; first full batch 2/3 complete (one subject correspondence conflict). Preserve this batch separately from final v7 experiments.

Ruling: Apply same-capture spatial constraints to confirmed identity references with the exact same observation ID and generation. v7 correctly separated camera groups but a subsequent subject review flipped a right-box ID onto the left-box group in the same frozen RGBD capture. The supplied depth was present; treating this as a soft historical location was incorrect. Different captures remain soft priors and include executed motion. Original v7 failure and v8 real-Luna replay (C6 to correct C4) are retained.

v8 local validation: 17 behavior checks, ruff F checks and compileall pass. Six frozen cross-camera comparisons plus the same-capture identity-flip replay pass with real Luna. Full v8 batch is tracked in data/runs/depth-association-20261115/experiments-v8.

Ruling from latest user instruction: table and container placement share measured arrival at configured release XYZ, actual empty gripper, and the same four-way Decisions question. Remove the table-only projected final-position square. Accepted direct and inferred container deliveries both retain historical membership without individual reassociation or CAD fitting; current visible contents enter anonymous TSDF. Keep container CAD and table cuboid. The intermediate v9 release-distance identity refinement is removed because completed contents do not need individual re-identification.

Validation added: direct and inferred receipts both defer completed contents; invalid receipts and explicit active manipulation do not. Both placement questions use the same execution evidence and no final-position square. Validate complete cluttered-container and unknown-object table transfers on final v10.

Final v10 validation: 17 depth checks and 6 unified-placement checks pass, with ruff F / compileall / diff checks. Real Luna frozen box replays 6/6 and same-capture temporal replay pass. Both complete API simulations succeed: cluttered box inferred_success and unknown-object table direct_success, with measured delivery and both arms returned. Post-container audit confirms historical ID retained, individual reassociation and CAD layer absent, current anonymous TSDF present, container CAD and table cuboid preserved. Deployment hashes and runtime version verified. See docs/depth-association-20261115.md and ignored experiment audits.
