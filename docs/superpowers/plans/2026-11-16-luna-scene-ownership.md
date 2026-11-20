# Luna scene ownership and collision model implementation plan

Execute inline under the existing repository constraints: no agents, worktrees, SDK edits, real robot motion, or new images. The user's seven explicit requirements authorize this follow-up and supersede the previous Base-owned physical scene design.

Goal: Luna exclusively configures physical simulation objects; Base's editor affects only its own twin. One Base address selects the same robot/RGBD interfaces for real or simulated hardware. Publish a checked robot collision model within 200–400 spheres, closely fitting the grippers.

Architecture: Simulation owns a neutral physical-scene API and compiles only its accepted configurations. Luna owns CAD selection, scene recipes and the editing UI. A separate Base extension forwards simulation administration requests without storing or interpreting the scene; the existing Base editor has no connection to this API. Robot and camera calls remain through Robot Bridge. Real/unknown Base modes reject simulation administration in both client and server.

## Constraints and review focus

- Common primitives: box, sphere, cylinder, capsule, ellipsoid; CAD keeps its library dimensions. Preset/custom colors, instance count, pose, mass, gravity and fixed/free state are explicit per object. No-gravity free objects use gravity compensation; fixed bodies remain separate.
- Configuration changes must not silently teleport moving robots, accept stale revisions, or turn scene truth into task evidence. Recompile preserves robot joint state and actuator bindings. Frames bind physical scene revision, independently of Base's viewer revision.
- Simulation must start with no Base process available; Base starts only after Simulation is ready; Luna starts only after Base is ready. Real hardware is started externally, then Base, then Luna.
- Canonical algorithm references in product configuration/commands/display have no suffix. Docker's unavoidable internal default tag is recorded accurately in raw inventory, not misrepresented as another image.
- Remove only verified guangyu-owned redundancy after checking all container references. Preserve real-platform image and other owners' resources.
- Sphere fitting uses pinned cuRoboV2-Fast public APIs. Check coverage, excess envelope, gripper full travel, representative self-collision/limits and FK. Recompute workspace after the final model changes.

## Work units

- [x] Inventory all images with owner labels, names, history and container references; audit remaining explicit numeric references. Record per-ID keep/remove decisions.
- [x] Budget and fit a 200–400 sphere candidate using existing fit/audit scripts and project-side coverage checks; preserve official SDK. Validate gripper motion and regenerate version-bound workspace using guarded build entry points.
- [x] Replace Simulation's Base scene polling with its own neutral scene store/API and generalized primitive/CAD compilation. Add physics tests for gravity, gravity compensation, fixed bodies, counts/colors and failed/stale updates.
- [x] Separate Base viewer scene from physical simulation. Add a minimal forwarding capability with mode and robot-idle checks; remove old physical-scene polling/publication bindings. Test that editing Base cannot change Simulation.
- [x] Update Luna's scene adapter/UI for primitive/CAD batches, per-instance color/gravity/fixed/mass settings, persistent recipes and automatic Base mode discovery. Test real-mode refusal before writes and isolation from task APIs.
- [x] Enforce ordered launch/readiness in existing service scripts/Compose; no implicit robot connection or motion on application startup.
- [x] Deploy only owned simulation services, verify source/deletion propagation, browser operations, RGBD revision/synchronization, SDK collision/FK results and finite command feedback. Publish actual image decisions and remaining measured model limits.

Tests remain outside Luna/Base repositories as required; Simulation/algorithm repositories retain their established tests. SDK versions, image IDs and task evidence are preserved in manifests and data/acceptance.

Results and measured limitations: [scene ownership audit](../../luna-scene-ownership-audit.md). Runtime and source propagation were checked against the final artifacts.
