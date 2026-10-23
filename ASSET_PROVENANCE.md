# Robot assets and MuJoCo adaptation

## User CAD pink replacement (2026-10-21)

The active `pink_wrapped` asset now comes from the supplied
`tripo_convert_891be992-2e91-4d83-b19a-dbbfcfa1acd1_flat_bottom_110mm.stl`.
Its SHA256 is `200868327093f29d38856dd769ed11855a0cab7f26e1d8aa72207e6cf0298f3f`.
The header declares millimetres, corrected orientation, flat base Z=0, and edits
below Z=4.5 mm. Measured native dimensions are 111.085350×62.787369×110 mm;
the AABB center is [0,0,55] mm. The visual retains all 10,924 triangles and
their winding, including 368 coplanar base triangles. The source is still not
watertight or consistently wound; no implicit material density is inferred.

`scripts/build_cad_assets.py --asset pink_wrapped` applies the existing single
millimetre-to-metre conversion/AABB centering and eight axial convex collision
bands. It retains the explicit .08 kg demo mass. The 61 source/generated blue
and black files and their manifest records remain unchanged. Old `source/pink.stl`
and existing content-hash render meshes remain for historical provenance;
only the replacement is selected by the current catalog. Re-encoding an old
CAD scene still depends on that recording's asset compatibility with the
current catalog; this change does not claim historical re-encoding validation.

Current default XY, 110 mm height, initial Z=.461 m, robot joints, cameras,
servo/contact settings and CAD libccd policy remain unchanged. Evidence and
source/asset preservation hashes are in `data/pink-flat-bottom/`; current
render and source fingerprints are in `resources/demo_observation_poses.json`.

`assets/marvin_urdf/` was copied on 2026-09-28 from the supplied local
`C:/project/tianji-simulation/assets/marvin_urdf/` portable snapshot. It contains
the camera-bearing Marvin M6 dual-arm robot and RAG100P grippers, including their
visual meshes, authored collision meshes, inertias, limits and mimic joints.
The copied source files are unchanged. This project does not import or resolve
files from tianji-simulation or tianji_base at runtime. Ownership and licensing
remain those of the supplied original assets; this note grants no new license.

`src/luna_mujoco/model_builder.py` is a project-side MuJoCo adapter derived from
the import procedure in the supplied `marvin/twin/mujoco_bridge.py`. It retains
the necessary OBJ group merge, distinct visual/collision mesh names, canonical
URDF-to-MJCF import, joint position/velocity limits, MuJoCo arm and RAG position
servos, and linear mimic constraints. The arm simulation capacity adjustment
below is separate from the unchanged authored URDF effort ratings.
Staged files live in a temporary directory while the
runtime exists. SDK and original-project source files are never modified.

MuJoCo is pinned to **3.10.0** in `pyproject.toml`. The adapter uses its public
`MjModel`, `MjData`, `MjSpec`, `mj_step`, `mj_forward`, `mj_fwdPosition`,
`mj_stateSize`, `mj_getState` and `mj_setState` APIs. References checked:

- [MuJoCo 3.10 API types and integration-state definition](https://mujoco.readthedocs.io/en/3.10.0/APIreference/APItypes.html)
- [MuJoCo API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
- Installed 3.10.0 public Python docstrings and headers; the actual adapter and
  state roundtrip are exercised with the installed package.

The browser mesh cache under `assets/render_meshes/` is generated from compiled
MuJoCo mesh vertices and faces, rounded to 0.1 micrometre. Meshes use the compiled
local coordinate system and therefore match each geometry's measured world
pose. It is a derived display artifact, not a different collision model.

Physics conventions:

- A = `Joint1_L` through `Joint7_L`; B = corresponding `_R` joints.
- Actual TCP frames are the preserved URDF `TCP_Link_L` and `TCP_Link_R` bodies.
  Fixed bodies are retained during import. Gripper mounts sit +7.5 mm along TCP
  Z with the source URDF's mount rotation; TCP is not the finger contact center.
- Robot self-collision is disabled, following the supplied bridge, because the
  source convex collision hulls overlap in the zero pose. Articulated arm and
  gripper contacts with scene objects and the fixed table remain enabled.
- The source base/stand convex hulls enclose part of the working volume, so
  base/stand contact masks are disabled as in the source bridge's scene mode.
  Their visuals remain. Generic scene objects retain full physical contacts
  throughout every grasp and never become welded, kinematic, or attached.
- The table is a world-body box with no joint. Scene `size` values are full
  dimensions and positions are centers. Gravity is -9.81 m/s² along Z.
- Arm actuators consume joint-angle position references in radians, with
  stiffness 240000 Nm/rad and velocity damping 600 Nm s/rad at 1 kHz using
  MuJoCo `implicitfast`. Their finite simulated actuator and total-joint
  capacities are 10 times the imported URDF effort: 1080/1080/660/660/180/180/180
  Nm per arm. This is the requested approximate rigid position mode for
  MuJoCo, not a hardware rating or a change to the source URDF. Native robot
  gravity compensation remains inside the same total-joint capacity.
  RAG gripper position servos retain stiffness 200, damping 10 and their
  original 8.066 Nm driver effort limit; object contacts and free-body dynamics
  remain enabled. Task code never commands torque/force and never writes robot
  generalized positions. Only initialization and baseline reset initialize
  robot state; mock unpacking changes only the selected object's free pose.
- Confirm saves `mjSTATE_INTEGRATION` and controller references. Reset restores
  the saved model, scene, integration state and controls, clears commands and
  increments the generation, invalidating work that returns late.

CPU verification: `python -m pytest tests/test_simulation.py -q`. These tests
exercise real MuJoCo gravity, motor motion, gripper servos and reset cancellation;
they do not by themselves establish an end-to-end learned grasp success rate.

The 2026-10-21 arm-capacity/stiffness adjustment follows the frozen actual
`2e473e106e50411d9004191ccf36b242` grasp. The original 12000 stiffness and
18 Nm wrist cap reproduced 2.008 degrees peak loaded-closure drift, with J7
saturated for 96.85% of sampled closure time. Increasing capacity alone still
allowed 0.510 degrees peak compliance. The new position settings reduce the
same replay to 0.03492 degrees peak at 1 ms, and 0.04370 degrees at 0.5 ms.
`tests/test_loaded_position_servo.py` checks every integration step for both
arms against the existing 0.05-degree simulator start tolerance, finite state,
unchanged desired angles, real finger contact and zero task-applied forces.
The runtime 1-degree start gate and 5 mm collision policy are unchanged.
The archive, one-variable comparisons and exact source hashes are retained in
`data/2e473-servo/`. This verifies a bounded contact/controller regression, not
the subsequent live lift or whole transfer.

## Simulation dependency boundary (audited 2026-09-29)

This project does not contain the Marvin physical-controller SDK or its driver.
The corresponding source-project files are
`tianji_base/extensions/adapters/marvin/driver.py` and its `vendor_sdk/`
directory, including `fx_robot.py`, `pointset.py`, `libMarvinSDK.dll/.so` and
`libKine.dll/.so`. None is present in this project or in the audited local
`deploy-bundle.tar.gz`. The audit parsed imports in all 46 production and
vendored Python files, checked dynamic-import calls and native-library loading,
and inspected the archive's 272 members. It found no import or native-library
load of the physical-controller SDK. No SDK files needed deletion.

The retained files have distinct simulation or algorithm roles:

| Files | Role and dependency |
|---|---|
| `assets/marvin_urdf/` | Robot geometry, joint limits, inertias and gripper model used by MuJoCo and cuRobo. The Marvin name identifies the robot model. |
| `resources/curobo/marvin_gui_dual.yml` | Project collision configuration derived from the original 38-link model. Its `tianji_base` path records provenance. Sphere arrays follow the reviewed project fit; kinematics, buffers, collision-pair exclusions and open-gripper locks are preserved. |
| `vendor/tianji_curobo/` (26 Python files) | Copied algorithm adapters, configuration, contracts and GPU service support. `geometry_service.py` imports `CuroboRuntime`; `backend_builder.py` imports planner and robot-configuration helpers. |
| `vendor/tianji_graspgen/` (7 Python files) | Copied inference client, service and data contracts. `algorithms.py` imports `GraspGenClient`; Compose starts the copied GraspGen service. |
| `resources/curobo/sdk-source-manifest.json` | Integrity manifest for the NVIDIA cuRobo algorithm SDK installed in the GPU image. It contains source hashes, not a robot driver. |

At the 2026-09-29 copy audit, all 33 adapter files and the original collision
YAML matched `docs/algorithm-provenance.json`. This records the original copy;
it does not claim that the current collision sphere arrays are unchanged.
The reviewed project refits replace sphere geometry without modifying the
original URDF/visual assets or the installed NVIDIA cuRobo SDK. See the current
collision-model section in `README.md` for the selected archive and validation.
References to the original `marvin/twin/mujoco_bridge.py` above describe the
MuJoCo adapter's origin; that source module is not imported at runtime. References
to an algorithm "SDK" mean NVIDIA cuRobo, whose installed source is left intact;
they do not mean the Marvin physical-controller SDK.
