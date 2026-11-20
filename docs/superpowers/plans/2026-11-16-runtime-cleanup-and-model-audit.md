# Runtime cleanup and model audit

User-authorized follow-up across Luna, Base, Simulation and the shared algorithm repository. Work in the existing checkouts, without agents or worktrees. Do not change SDK sources, other users' services or the real controller.

## Intended behavior

- One canonical unversioned image name per algorithm; Docker's implicit latest tag is an implementation detail. Preserve exact SDK versions and image IDs in manifests. Inspect references before removing retired guangyu containers/tags. Keep jodell-console:0.2.0 for the platform.
- Responses annotates task entities and necessary supports in each visible camera. Decisions validates those labelled images before identity registration. Boxes do not replace masks or RGBD geometry.
- Put CAD selection and manual simulation setup in a separate Luna configuration surface. Base accepts a generic scene and geometry through its scene capability; it has no dependency on the CAD library. Task algorithms cannot read this scene truth. Recording remains in Luna. The user explicitly selected the Luna editor and CAD entry point.
- Remove GraspGen approach-axis direction filters and saved/UI options, including repeated endpoint direction restrictions. Preserve jaw symmetry, actual-vs-requested FK checks, collisions and the vertical Lift path policy.
- Remove the robot bridge's required Action Stream capability. Finite trajectory, joint and gripper commands retain adapter-level arbitration, cancellation and feedback checks.
- Reuse the verified cable-free gripper geometry across URDF consumers. Compare semantic joint definitions, camera transforms, collision coverage and workspace provenance; adjust only geometry shown to need adjustment.

## Execution and acceptance

1. Snapshot image/container references and robot geometry. Capture focused failing regression probes outside Luna/Base repositories.
2. Implement direction-filter removal and bridge decoupling. Run numeric and provider-boundary probes.
3. Implement the CAD-independent scene boundary and Luna editor. Verify that task observation paths still use only RGBD and measured robot state.
4. Audit/update the URDF, camera collision coverage and workspace. Use pinned cuRobo public interfaces for numeric FK/IK verification.
5. Update all deployment references/build policy/manifests to canonical algorithm names. Tag existing images, recreate only owned containers as needed, then remove confirmed redundant owned resources without deleting data volumes.
6. Run appropriate static/build tests and isolated simulation integration checks. Record exact evidence and remaining limits in data/acceptance/runtime-cleanup-model-audit.
