# Luna CAD scene, cameras and fixed observation pose

The project is `C:/project/tianji-luna-mujoco` and
`/home/guangyu/projects/tianji-luna-mujoco`. Robot forward is +X, left is +Y.
All runtime lengths are metres. The user's 2026-10-21 CAD files replace the
primitive bins and pink/black objects; only tables retain primitive geometry.

## Current scene and asset boundaries

`resources/demo-config.json` and `demo_scene(object_count=None)` default to one
pink wrapped CAD object and retain convenient count 1–4 presets. The editor can
add any number of instances; an existing user-edited scene takes precedence.

- Main table: fixed cuboid, size [0.60,0.80,0.08] m, center [0.572,-0.026,0.532] m,
  top Z=0.572 m, white by default. No small tables.
- Blue CAD bins: native size [0.410,0.310,0.145] m, fixed in the world with collision.
  Default centers are [0.481,0.573,0.571] and [0.481,-0.625,0.571] m, symmetric
  about table Y=-0.026 m; default yaws +90/-90 degrees. Their complete cavity,
  tapered walls, rounded corners, ribs and lip come from CAD.
- Pink CAD: size [0.111085350,0.062787369,0.110] m, yaw 180 degrees, with gravity.
  The one-object preset uses offset [0,-0.08] m from the left-bin center; the pair
  uses [0,-0.08] and [0,0.08] m. Z follows the CAD inner floor plus half object
  height and a 1 mm settling gap. Other preset offsets are defined in the config.
  These scene-authoring coordinates never enter runtime visual perception.
- Black CAD: native size [0.044,0.056,0.0745] m. The mock rotates local Y by 90
  degrees so the part lies flat, changes yaw by 30 degrees, and outputs center
  XY=[0.56,-0.22] m. Z is tabletop+0.023 m, leaving a 1 mm settling gap; gravity
  remains enabled. Decisions and current RGBD must independently confirm it.
- The invisible square table target is 0.32 m wide, center XY=[0.50160,0.11646] m.
  Release XY refers to the gripper front, not the held point-cloud centroid.


`cad_assets.cad_catalog()` returns the UI asset list, dimensions, default
colors and declared demo masses; `cad_asset(id)` returns copied provenance and
shape priors. Asset IDs are `blue_box`, `pink_wrapped`, `black_core`.
A CAD instance uses `kind="cad"`, `asset_id`, centered AABB `position`,
`yaw_deg`, named or #RRGGBB `color`, and explicit `mass_kg`. Its size is fixed
by the native asset, not distorted to the obsolete primitive dimensions.

The copied originals, generated visuals/collision meshes, source transform and
hashes are under `resources/cad/`; see its README and `scripts/build_cad_assets.py`.
The STEP declares mm, as do both STL headers. Scale and AABB centering happen
once. Source triangle geometry/winding is retained. Pink and black are not
watertight; their signed source volumes are not treated as material volumes.
The current pink is the user's `tripo_convert_891be992-2e91-4d83-b19a-dbbfcfa1acd1_flat_bottom_110mm.stl`,
with source AABB center [0,0,55] mm and 368 authored triangles on flat Z=0.
The superseded `source/pink.stl` is retained for provenance only. Blue and black
asset files and records are byte-for-byte unchanged by this replacement.
The bin uses 49 cavity-preserving convex pieces; pink/black use eight axial
convex pieces each. Local concavities/threads may be filled in these declared
rigid collision approximations. Demo masses are 0.5/0.08/0.06 kg respectively,
with convex-piece volume shares for inertia. They are not CAD density claims.

`demo_region()` retains the unmarked 0.32×0.32 m square at XY=[0.36,0],
with bounds X=[0.20,0.52], Y=[−0.16,0.16]. Runtime support height comes from
measured RGBD. The sequential unpack task uses this shared region without slots.
Older workspace and primitive-grasp audits are historical position screenings,
not validation of the new CAD contacts, tool pose or trajectories.

## Camera source and frame conversion

The three user files are copied byte-for-byte under
`resources/cameras/user-extrinsics/`. Their original source is
`C:/Users/l1ght/Downloads/camera_{left_wrist,head,right_wrist}_extrinsic.json`.
Every copied file has a SHA256 recorded in `resources/cameras/calibration.json`.
Raw calibration samples are retained in those source files and are not printed
by the import script.

`scripts/import_camera_extrinsics.py` verifies camera ID, `base_frame=base_link`,
matrix shape, rotation and `translation_unit=mm`, then scales the matrix's
translation by 0.001 exactly once. Rotations are unchanged. Wrist files declare
`T_flange_cam` and attach to `TCP_Link_L` / `TCP_Link_R`. Head declares
`T_base_cam`; it is already expressed in robot world/base_link and attaches to
`world`. Its translation is
[0.1273603218756391,−0.0034876335486242415,1.199981040227405] m.
The accompanying `T_world_base` is not multiplied into that result.

The supplied files contain extrinsics and an intrinsic revision identifier,
not replacement numeric RGB intrinsics. Existing registered RGB intrinsics and
their source provenance are retained. The current calibration ID is
`user-extrinsics-5a390ce34b936bd8`.

## Fixed joints and current offline evidence

Left A / camera 2 remains exactly
`[-30.116,55.982,21.322,-109.932,14.629,-29.333,-60.589]` degrees.
Right B / camera 1 is the FK mirror
`[30.116,55.982,-21.322,-109.932,-14.629,-29.333,60.589]`.
The 128-sign URDF correspondence check is preserved; no pose optimizer changes
these angles. `resources/demo_observation_poses.json` fingerprints the current
factory, CAD manifest, model adapter, robot and calibration.

Actual local MuJoCo RGBD/segmentation at those joints is archived in
`data/edge-alignment-audit/observation/` for presets 1–4. Every pink object has
at least 100 valid-depth pixels and there are zero initial robot/scene or pink
contacts. The current default pair has 2945/3598 visible pixels in camera 2;
counts 1/3/4 have 3086, 3093/2909/3147 and 3055/3126/2972/3073 respectively.
The default RGB was inspected with the robot geometry fully rendered. Both
pink bodies are visible and the near object has a small gripper occlusion.
All eight bin-envelope corners are now inside the image, with a 36.894 px
minimum edge margin; full envelope coverage remains diagnostic, not an
additional acceptance gate. `data/edge-alignment-audit/cad-envelope-clearance.json`
records actual STL bounds, rounded-cavity margins and edge alignment. These
checks ran serially on one logical CPU with numerical-library threads set to 1.

Historical evidence at the earlier X=0.20 bin layout is under
`data/pink-flat-bottom/observation/` (the default pair had 3004/3102 pixels).
The previous source's bounded comparison at the same joints changed only the second object's
position from [0.16,0.64] to [0.24,0.63]; approximate visible/projected silhouette
coverage improved from 50.3% to 92.5%. The first remains about 93.3% visible.
`data/cad-inspection/visibility-comparison/` preserves the original and both
candidate images, triangle-projection denominator and actual masks; raster
edges make these percentages approximate. All robot geometry remains rendered.

The earlier bin projection audit measured about 91% of its unoccluded triangle
projection inside the frame, with the near edge clipped. That clipping result
does not describe the current translated bins. The fixed-pose checks do not
establish collision-free movement or execution-arm reachability.
The current endpoint sphere audit uses conservative AABBs around each CAD
collision piece plus actual MuJoCo contact checks. Runtime planning remains
responsible for the measured map and complete trajectories.
The new pink's full native X width (111.085 mm) exceeds the URDF's 103.001 mm
fully open pad gap; its Y width (62.787 mm) fits. This is only a directional
width prerequisite for candidate filtering, not evidence of a valid grasp.
The replacement's earlier-layout 100-second actual CPU settle audit kept both pinks below
the existing speed gate throughout seconds 5–100 (angular-component peaks
.001993/.001175 rad/s); public `confirm()` succeeds. Settled three-camera RGBD
is archived in `data/pink-flat-bottom/settled/`. It validates that earlier layout's
initial support and observation, not the new alignment, learned grasp or transfer
success. No long settle or new motion was run for the edge-alignment change.

`data/cad-inspection/measured-bin-fixtures/` supplies actual frozen left/right
RGBD, offline segmentation masks and world point clouds for independent CAD
registration tests. Pose ground truth is separate and test-only. The fixture
cannot become a runtime pose source. Historical primitive figures are retained
under `data/demo-observation-poses/`, `data/mock-unpack/` and earlier layout
folders; their geometry/pixel counts are not current CAD evidence.

## Transactional mock unpacking and replay

`Simulation.mock_unpack(work_area, output_pose=None, task_id=None, *, generation,
target_id=None, owner_id=None)` is simulator actuation, not visual evidence.
It requires idle commands and exactly one supported pink wrapped entity inside
the work-area footprint. Canonical IDs are receipt-only; physical selection uses
the simulator's geometric transaction boundary. The physical body/free-joint ID
is retained, while the perception layer separately owns semantic entity lineage.

The output is `kind="cad", asset_id="black_core"`, size 44×56×74.5 mm and
explicit demo mass 0.06 kg. Its default center is
[0.36,-0.24,table_top+0.03825] m, hence Z=0.59825, with yaw changed by +30°.
This places the complete CAD inside the right-wrist image at the fixed observation
pose, while retaining a head-camera view at the checked natural VLA pose.
Custom output still requires changed XY/yaw, upright tabletop support and the
whole footprint inside the right half of the desk. Collision penetration beyond
1 micrometre rejects the transaction; this integrity check does not alter the
runtime 10 mm planning clearance.

The new model compiles before atomic state replacement. Joint/body/actuator and
integration layouts must match. Public `mj_getState`/`mj_setState` copy current
integration state; only the target's free pose and velocity change. Robot
positions/velocities/controls, other objects, simulation time, generation,
controller targets and reset baseline remain intact. `configure()` is not used.
The receipt remains `source="mock_vla", is_visual_evidence=false`, with actual
completion timing, commanded pose/size, physical ID and manifest revisions.

Manifest revisions invalidate old sensor results. Visible CAD rows expose
`asset_id`, `object_size_m`, compiled `local_position_m` and `local_quaternion`
(wxyz), so editor previews preserve MuJoCo's mesh-axis offset. Mesh URLs are
content-addressed, so changing mesh indices cannot overwrite prior replay
geometry. Tests verify old mesh bytes survive the swap and no mixed-revision
RGBD capture publishes afterward.

## Position servos retained

### CAD contact stability and historical irregular base (2026-10-21)

The CAD-only model uses MuJoCo 3.10's public `nativeccd=disable` flag
(libccd); multiCCD remains enabled. Primitive-only historical scenes retain
the original native collision pipeline. This contact-policy change did not alter
CAD geometry, masses, contact coefficients, timestep or servo settings. The
later user-supplied flat-bottom replacement retains the same policy. No
qpos/qvel reset or sleeping is used to settle objects.

The initial Ubuntu scene with the superseded irregular-base pink exposed persistent rocking. An isolated
100-second CPU replay reproduced a .03935 rad/s peak after five seconds and
only .021 s continuously below the existing .015 speed gate. Contacts were
between the pink lowest convex band and the blue bottom, not between all eight
bands. Native multiCCD was already enabled. Disabling NoSlip or tightening
native collision tolerance to 1e-9 did not resolve it. The libccd comparison
kept both pink objects below .015 throughout seconds 5–100, with peaks
.01250/.01101 rad/s. A black CAD resting on the main table and a second black
CAD dropped naturally into the right box also remained below that gate for
95 seconds; this validates natural support dynamics, not a robot drop task.

`tests/test_cad_settling.py` checks every step in a late 15-second window,
and loaded A/B gripper regressions exercise both collision pipelines at
1 ms and .5 ms. Evidence is in `data/cad-inspection/settle-diagnostic/`.
The gate remains .015 for .25 s; simulation contact stability is separate
from task placement tolerances. See the official
[MuJoCo 3.10 collision flags](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#option-flag-nativeccd)
and [multiCCD description](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#option-flag-multiccd).

Existing loaded closure coverage is retained and new A/B public `set_gripper`
tests use a frozen real grasp plus its
FK-symmetric counterpart: both arms keep identical desired q and ctrl, actual
finger/object contact occurs, and measured arm error stays below the existing
one-degree physical regression bound. This is a test bound, not a new task
threshold or a claim of zero mechanical compliance.

The later actual two-object run `2e473e106e50411d9004191ccf36b242` exposed a
stronger contact load: wrist J7 saturated at the unchanged URDF's 18 Nm rating
and drifted 2.6046 degrees online. Frozen replay reproduced the failure before
the fix. The MuJoCo approximate rigid position mode now uses arm stiffness
240000, damping 600 and finite simulation capacity 10× authored arm effort;
gripper gains and authored 8.066 Nm limit are unchanged. This is a simulator
position-control approximation, not a new hardware rating or task torque mode.
The full actual approach/grasp replay plus loaded closure stays below the
unchanged .05-degree simulator start tolerance at both 1 ms and 0.5 ms, with
unchanged desired q/ctrl and no qpos overwrite. See `ASSET_PROVENANCE.md` and
`data/2e473-servo/`; subsequent live lift/transfer still needs execution evidence.
