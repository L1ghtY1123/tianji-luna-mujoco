# Robot assets and MuJoCo adaptation

`assets/marvin_urdf/` was copied on 2026-09-28 from the supplied local
`C:/project/tianji-simulation/assets/marvin_urdf/` portable snapshot. It contains
the camera-bearing Marvin M6 dual-arm robot and RAG100P grippers, including their
visual meshes, authored collision meshes, inertias, limits and mimic joints.
The copied source files are unchanged. This project does not import or resolve
files from tianji-simulation or tianji_base at runtime. Ownership and licensing
remain those of the supplied original assets; this note grants no new license.

`src/jev_mujoco/model_builder.py` is a project-side MuJoCo adapter derived from
the import procedure in the supplied `marvin/twin/mujoco_bridge.py`. It retains
the necessary OBJ group merge, distinct visual/collision mesh names, canonical
URDF-to-MJCF import, joint effort limits, torque PD, RAG position servos and
linear mimic constraints. Staged files live in a temporary directory while the
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
- Arm motor references are integrated at 1 kHz using the source bridge's
  bounded torque PD. Trajectory execution never writes robot generalized
  positions. Only scene configuration and baseline reset initialize state.
- Confirm saves `mjSTATE_INTEGRATION` and controller references. Reset restores
  the saved model, scene, integration state and controls, clears commands and
  increments the generation, invalidating work that returns late.

CPU verification: `python -m pytest tests/test_simulation.py -q`. These tests
exercise real MuJoCo gravity, motor motion, gripper servos and reset cancellation;
they do not by themselves establish an end-to-end learned grasp success rate.

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
