# Held release height implementation

Approved scope: numerical endpoint search only, no deployment, GPU probe or robot motion.

1. Add failing tests for clear low goals with failed held IK, achieved FK above the configured object-floor gap, and workspace clipping at the old 20mm limit.
2. Add a project-only release-height search module for shared sampled bounds, exact geometry/feedback bindings, public batch IK selection and proof validation. Keep the old geometry-only function explicitly compatible.
3. Wire production DemoAlgorithms and GeometryRuntime to the same search descriptor. World queries finish before acquiring the existing IK GPU operation; each sample is evaluated once. Preserve independent world/IK evidence and all original XY candidates.
4. Validate selected and actual-release full CAD/OBB gaps, cache freshness and trajectory separation. Add route/lock and tampering regressions using CPU fixtures; no SDK edits.
5. Run related placement, carrying, geometry-service tests and Ruff. Update README, guide and resolve_placement/plan_place contracts. Parent owns full deployment and real validation.
