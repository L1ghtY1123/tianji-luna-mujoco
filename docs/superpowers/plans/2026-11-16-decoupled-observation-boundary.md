# Measured robot boundary, display and runtime image consolidation

The user authorized implementation and removal of unused owned images. Work remains in the existing four repositories, without subagents or worktrees. No real robot commands.

1. Remove the local gripper-width predictor and unused destination-configuration screening. Keep measured Base gripper feedback, cuRobo planning and joint-based FK validation. World geometry must come from RGBD and registered model priors, never Simulation object state.
2. Define explicit robot gateway/geometry contracts. Compose BaseClient and the algorithm client only at the application entry point. RobotRuntime uses those interfaces; recordings use public asset access, with no HTTP/client internals outside the adapter.
3. Make the measured Base embed publish only the robot and measured joints/grippers. Base's normal console stays colored with its own scene editing. Luna uses a neutral display option and overlays only its validated observation/collision scene, sharing the same robot renderer.
4. Simulation's own render materials use gray-white robot/column and black grippers/cameras, without changing Base model assets, geometry or calibration.
5. Retag the seven current runtime images as tianji-{luna,base,simulation,algorithm-curobo,algorithm-sam3,algorithm-graspgen,algorithm-foundationpose}:2026.11.16-guangyu. Record provenance and exact IDs. Update launch configurations, restart only owned idle services, verify health and RGBD. Delete only confirmed owned, unused images/tags, retaining shared and stopped-container dependencies. No global prune or image build.

Verification: temporary contract probes (repository prohibits local test directories), Simulation material regression, frontend build, Base measured-scene exclusion checks, browser three-camera/robot/overlay checks, simulation trajectory and cleanup, image/container before-after audit, SDK integrity. Report real-hardware and full-task limitations separately.

## Completed verification

- RobotRuntime depends only on injected RobotGateway/RobotGeometry; four boundary checks and a pure substitute implementation passed. The nine transport/feedback probes passed after moving immutable observation contracts and normalizing execution receipts at the adapter.
- Removed gripper-width prediction and unused destination configuration screening, including evidence serialization and legacy documentation. Existing FK endpoint and collision checks remain joint/model based.
- Actual Base measured scene has 45 robot nodes and an empty environment; it does not call the editor snapshot. Browser validated neutral robot rendering and three live cameras. Raw and collision overlay fixtures stayed in browser interception; rendering and clearing passed without JavaScript errors. Matrix refresh and buffer geometry serialization preserve placement and support line geometry across the two render modules.
- Simulation palette/model tests passed. The broader five-module run passed 29 tests and four subtests, exposing two obsolete visual assertions (old box housing and shared Base/simulation colors). Updated these assertions to the current public STL contact model and separate palettes; all three visual tests then passed.
- Algorithms: 52 checks, 50 pass and 2 CAD-dependency skips. The five build-command tests passed after updating canonical image names and overwrite protection.
- Real isolated simulation: continuous trajectory with measured close/open receipts; both arms disabled afterward; complete RGBD and official RobotSegmenter; two archived viewer frames rendered without MuJoCo. No real hardware or full unpack task acceptance.
- Seven current containers use canonical tags with unchanged image IDs. Optional SAM6D and the retained old cuRobo build/base-planner runtime have explicit canonical aliases. Deleted seven unused owned images and two superseded aliases. Kept container-referenced/shared images and shared builder caches. No new builds.
- SDK integrity: 485 official files at cuRobo 0.9.0 / 42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83 unchanged.
- Evidence: data/acceptance/decoupled-boundary-20261116; full tag/ID/source manifest: deploy/runtime-images.json. Deployment uses read-only source mounts; canonical image names identify the existing runtime environments.
