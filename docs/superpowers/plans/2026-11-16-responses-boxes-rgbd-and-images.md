# Responses boxes, RGBD display and shared images

The user authorized these changes. Implement inline in the existing repositories without subagents, worktrees, new image builds or real robot motion.

## Design

Responses returns task-relevant physical instances with normalized per-camera boxes on the same frozen three-camera capture. It omits invisible views and unrelated objects. One ref represents one physical instance across views; reference_camera_id selects its clearest view. The application draws these boxes on the original RGB, preserving the pixels and capture identity. Decisions confirms the proposed identity and cross-view correspondence, or returns rejection/unknown. Historical identity comparison still precedes permanent ID allocation. Boxes remain visual references, not silhouette masks or object geometry. Subsequent perception may use SAM3 to obtain the actual masks required by FoundationPose and depth mapping.

Simulation lighting is explicit and stable for all cameras. Base's MuJoCo provider implements the existing separate RGB, depth-preview, depth16 and pointcloud modes. Raw depth stays uint16 millimetres; previews are display-only; pointclouds deproject the same registered RGBD and carry frame/sequence/calibration metadata. The existing camera dock and pointcloud viewer consume the provider without direct Simulation access.

Images use semantic versions: shared algorithms have no user/date suffix; the real console remains jodell-console:0.2.0. Private Luna/simulation/console aliases may contain guangyu. Existing IDs are retagged, references are updated, and only superseded unused aliases are removed. Remove SAM6D image and its stopped retired owned container, preserving source, weights and data volumes. Do not alter unrelated containers.

## Implementation and checks

- [x] Initial boxes: EntityProposal schema, Responses prompt/schema, bootstrap and registrar, dated reference rendering and observation overlays. Nine temporary checks cover three-view boxes, invisible/invalid views, duplicate refs, task-only scope, rejected/unknown Decisions, historical identity conflicts workspace filtering and live camera labels without recording. Initial SAM3 preparation removed; no box-derived object geometry.
- [x] Camera provider: depth16/preview distinction, calibrated XYZ+RGB bytes, stale/disabled frames, dictionary capability arguments and explicit live/stale/stopped camera states. Four temporary checks and the real Base camera dock/pointcloud capability passed.
- [x] Lighting: explicit scene headlight improved the neutral plane from about 88 to 130 RGB levels, without clipping or raw depth changes. Nine simulation camera/model checks passed. Public MuJoCo API used; SDK remains read-only.
- [x] Naming/cleanup: current Compose/build commands/manifests/docs and semver validation updated. Retagged existing IDs without builds; removed SAM6D and eight superseded date aliases. Real console alias stays unchanged.
- [x] Integration: scoped synchronization and owned-service restarts completed. Browser RGB/depth/pointcloud and Luna preparation passed without JavaScript errors. Actual Responses/Decisions requests, frontend build, static checks and algorithm checks passed within the boundaries below.

## Evidence and limits

Evidence: `data/acceptance/responses-rgbd-images/`; image IDs: `deploy/runtime-images.json` and `../tianji-algorithms/manifests/deployed-images.json`. Seven owned services verified; five Luna/algorithm services retain the total four-CPU limit. No MuJoCo/GLFW/OpenGL packages in the Luna runtime. Fifty algorithm checks passed; two CAD dependency checks skipped locally. Official SDKs unchanged.

Live supported sample: “把桌面上的绿色方块放入蓝色盒子。” One Responses call generated task boxes plus a validated graph; three Decisions calls each received all three boxed RGB views and registered the green cube, blue bin and required table. No initial SAM3 call or robot motion; joint deltas were zero. Invisible wrist-view subjects were omitted. Earlier sample with requested initial return posture conflicted with configured observation return posture and was explicitly unsupported; its separate localization evidence is retained. Full grasp/place execution and the user's full collection of side/containment/ordinal examples were not tested in this turn.

Browser checks toggled RGB/depth/pointcloud for every camera and verified real 3D pointcloud rendering and world poses. Approximately 278k/271k/307k points were returned; reconstructed floor heights are within 0.6 mm of zero. The existing hover toolbar requires hovering before clicking in browser automation; no UI workaround was added. Camera live-status reporting was corrected at the provider.

Shared stopped containers still reference some historical algorithm aliases, so those aliases remain. SAM6D source, weights and data remain; its runtime image, tags and retired owned container are removed. Two unrelated temporary containers disappeared during the session; the audit records that observation without attributing it to this work. No commands targeted those containers.

## Review focus

- A rectangle includes background: never store it as an instance mask or measured object centroid.
- A proposed view can be occluded or wrong: Decisions must not treat the proposal as established identity.
- A historical box is tied to its original capture: do not project it onto a new observation.
- Disabling a depth preview must not silently corrupt pointcloud mode; report disabled/stale data honestly.
- Shared image names must not overwrite an existing different image ID; stopped containers still count as image references.
