# Independent algorithm services

`Algorithms` is a synchronous, SI-unit client boundary. The app calls it in worker
threads while MuJoCo keeps stepping. Defaults are `LUNA_CUROBO_URL=http://127.0.0.1:8310`
and `LUNA_GRASPGEN_URL=http://127.0.0.1:8323`. There is no production mock or fallback.

Start the dedicated services from the new project:

```sh
docker compose -f deploy/algorithms.compose.yaml up -d
PYTHONPATH=src .venv/bin/python scripts/validate_algorithms.py
```

Only the two named `jev-mujoco-*-guangyu` containers and their project volume are
managed. Existing containers and original project directories are untouched.
The copied source, robot meshes, original collision YAML and source manifests are
all inside this project. Runtime does not import GaP or either original project.
The GPU images are reused without rebuilding or changing their SDK installations.

## Geometry and frame contract

`geometry.py` is the simulation geometry component. It caches a deterministic,
4096-point surface sample per primitive kind and full extent. Pose changes reuse
that local cloud. Observation reports object metadata and a cloud hash; it does
not put arrays in the 15 Hz state feed. Generation transforms scene samples into
the selected object's coordinates and passes the existing object cloud to the
unchanged GraspGen client and service.

The legacy wire names `scene_points_camera_m`, `object_points_camera_m`, and
`T_camera_grasp` remain unchanged for compatibility. Their actual coordinate
frame in this application is explicitly `object:<id>`. The application stores
returned matrices as `T_object_grasp`. No camera, image, calibration, depth or
segmentation metadata is fabricated. GraspGen's official outlier pass, centering,
model identities, scoring and original candidate indices are unchanged.

The project calls that existing service at most twice, sharing the same cloud,
then globally sorts and retains at most 200 real candidates. A run-local `index`
identifies each retained candidate; `source_batch` and `source_index` preserve
the original service identity. The response reports actual count and both call
durations. The current service internally samples 200 and returns top 100 per
call; its immutable implementation is not changed. A short output is never
padded with fabricated candidates.

Arm A uses the source 180-degree rotation about grasp Z; arm B uses zero degrees.
Both are evaluated separately. The exact composition is:

```text
T_world_tcp = T_world_object @ T_object_grasp @ Rz(arm_orientation)
              @ T_grasp_ee @ inverse(T_tcp_ee)
```

The source Robotiq inference profile has `T_grasp_ee.z=0.195 m`; the source RAG
tool default has `T_tcp_ee.z=0.185 m`. The robot TCP is the imported URDF's
`TCP_Link_L/R`, also used by cuRobo. The imported MuJoCo pad geometry confirms
0.185 m lies within the pad's Z span during closure. This is a documented source
configuration, not a claim of hardware calibration.

## Known geometry ESDF

`POST /geometry/build` accepts only known primitive scene geometry and the selected
target identity. It builds exact signed distances for boxes, spheres and cylinders
at 0.01 m voxel spacing, using the official pinned voxel-center and sign convention.
It uploads separate public `curobo.scene.VoxelGrid` layers for the table and every
object. Each local grid has 0.30 m padding per side, exceeding the largest original
robot sphere radius (0.088 m) plus the 0.03 m activation distance.

The adapted user-owned `GeometryRuntime` installs all layers, then disables only
`semantic_instance::<selected_id>`. The selection is held through the entire
native `plan_grasp` operation, including approach, grasp and lift. The table and
every other object remain enabled. This never changes MuJoCo contact flags.
Generation and content hashes bind each scene; stale scene tokens are rejected.

The three native trajectories retain the actual SDK sample period and radians.
Joint names are reordered explicitly to `Joint1..Joint7_L/R`. JEV chooses between
the separately evaluated arms. Every reachable candidate of the chosen arm is
submitted to `plan_grasp_batch` in groups of at most 16. After all batches finish,
the highest original GraspGen score among fully feasible candidates is selected.
There is no serial single-candidate planning fallback.

## IK calls and arm selection

For N generated grasps, `evaluate_arms` sends `ceil(N/100)` independent IK requests
for A and the same number for B. Each request calls public `solve_pose` once,
with seeds handled inside the batch. JEV receives reachable candidate counts,
each candidate's score, position/rotation error, and the Euclidean joint distance
from the frozen evaluation start. This distance describes one representative IK
solution, not trajectory length or execution time. Arm selection has no fixed
weighted score formula; JEV chooses from arms with nonempty candidate sets.

The official pinned `BatchMotionPlanner.plan_grasp` performs four `plan_pose`
operations: goalset validation, approach, grasp, and lift. Each operation calls
internal IK per attempt. Therefore N=100, one planning batch, one attempt and no
early failure normally means two external IK solves plus four internal IK solves.
The extra goalset check does not produce a fourth executable trajectory.

For the current 200-candidate pool, external IK makes four requests, each with
100 targets. A100 and B100 fit the existing total cached IK capacity of 200;
switching to A200/B200 would repeatedly evict the other arm's solver. Planning
still processes every reachable candidate in groups of 16. With K reachable
candidates, it makes ceil(K/16) planning calls. Each fully executed planning call
has four internal solves with default one attempt. More reachable candidates
can increase total time despite preserving the cached cost of each batch.

All three collision paths (IK, trajectory optimization and dense hard clearance)
use original YAML spheres plus the original per-link buffer, with both gripper
drivers locked at 0 radians (open). No closing-motion envelope is requested.
The YAML, URDF and vendor sources are unchanged. `/api/collision-spheres` uses a
separate MuJoCo data instance, locks driver and mimic joints to the same open
configuration, and displays all 1362 spheres. It never modifies live physics.

If both arms return zero candidates, choosing `choose_arm` raises
`no_reachable_arm` before another model request. Planning and execution do not run.
One empty arm is excluded from JEV's arm options. Zero successes are a result of
the finite search and configured constraints, not proof of mathematical
unreachability. The current demo does not automatically regenerate grasps or
switch arms after all three-phase plans fail.

## Measured GPU validation, 2026-09-28

The installed wheel omitted Git metadata. Startup therefore compared all 361
packaged Python source files against SHA256 values from official commit
`8e734f3ced1df898990bcd92de40abce475907db`; every file matched. The original
collision YAML remained byte-identical: 38 links, 1362 spheres. Image IDs and
all copied wrapper hashes are in `algorithm-provenance.json`.

On the Ubuntu RTX PRO 6000, an independent continuously stepped MuJoCo scene
produced these real results:

| Operation | Result | Wall time |
|---|---|---:|
| Geometry ESDF build | Table and three object layers; selected target excluded | 1.24 s |
| GraspGen | 100 original candidates | 0.55 s |
| Separate A/B IK | A: 32 reachable; B: 31 reachable | 55.13 s cold |
| A three-phase plan | Original candidate 78; 61/41/41 samples | 40.63 s |
| B three-phase plan | Original candidate 11; 61/41/41 samples | 114.32 s |

Full inputs, matrices, returned indices, plans and timings are saved in
`algorithm-gpu-validation.json`. These checks prove inference and planning;
physical grasp success is assessed separately by the app's execution verifier.

Six CPU adapter tests verify signed distances, cloud reuse, arm frames, original
candidate indices, stale binding rejection and target-only layer disabling.
