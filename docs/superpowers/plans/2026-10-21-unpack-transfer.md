# Sequential unpack transfer implementation plan

> Implement in the existing Windows and Ubuntu projects. The user's explicit workflow is the design and authorization; no additional design approval is pending.

**Goal:** For each selected pink wrapped item, left grasp → shared table center → simulated VLA unpack → right grasp → drop into right bin → visual confirmation.

**Architecture:** Responses generates one bounded typed skill program and freezes canonical item IDs with both executing arms. A lifecycle ledger retains each ID across the pink-to-black appearance change; only right-bin confirmation completes it. Decisions verifies visual transitions; measured RGBD supplies geometry only at the prescribed dual-arm observation pose.

**Stack:** Python, MuJoCo position actuators, public cuRobo motion planning, SAM3, GraspGen, Responses/Decisions, existing browser replay.

## Constraints

- Two initial pink objects, separated inside left bin; fixed shared numerical table region, no slots.
- Pink 10×6×4 cm, black 6×4×3 cm upright with changed XY/yaw after mock VLA.
- One latest TSDF/ESDF; 2 cm voxels; 5 mm collision activation; first measured table cuboids reused.
- No sensor remapping outside dual observation pose. Explicit action-derived occupancy bounds are distinguished from measured mapping.
- GG tool Z within 45° of down, away from robot, one +1 cm tool-Z insertion, then batched IK and plan_grasp_batch.
- Grasp verification after lift uses gripper state. Attachment 24 spheres by default and strictly fewer than32, same-gripper overlap allowed.
- Right-bin transport uses attached public plan_pose, lift orientation, generous above-rim Z, configured quintic/time scaling; open/drop at goal.
- Mock VLA is recorded as simulated actuation and must never count as visual proof.

## Implementation and verification

- [x] Lifecycle/program: extend typed skills and initial Responses output with right bin and per-item arms; test partial completion, same ID, unsupported order and unknown transitions.
- [x] Perception: verify unpacking before SAM using the same observation capture; phase-aware queries and optional measured-consistent10/6cm priors; retain completed right-bin obstacles.
- [x] Planning: attached plan_pose transport, table-source clearance, remove old slot heuristics for black stage; verify binding,5mm and command interpolation; migrate sphere settings explicitly.
- [x] Simulation: transactional pink-to-black replacement with unchanged other state and generation; two-object layout; model revision in live and archived replay; gripper-close servo tests.
- [x] Runner: shared center table placement; intermediate verify does not complete; VLA dual pose, mock, retained-map return, unpack Decisions, observation-only remap, right pickup/drop, final verify.
- [x] Documentation/UI: current skills/contracts, guide, README, default instruction and replay stage labels agree with actual registry.
- [ ] Integration: focused tests then complete project/frontend checks, exact local→Ubuntu sync, actual two-object simulation with model calls and archived image/pointcloud/video evidence.

## Review focus

Unknown identity must not publish geometry; previous right-bin items must not count as the current drop. Old pink clouds/plans must expire at appearance change. A mock action must not reset simulation or other objects. Returning after mock must account for bounded changed occupancy without claiming new RGBD mapping. One object's table release must never advance the next object before its right-bin confirmation.

## Latest CAD and identity revision — supersedes earlier dimensions, 5mm and shared entity ID

The user's latest request authorizes CAD replacement, a 10mm collision threshold, symmetric gripper coverage, unrestricted manual scene instance creation, configurable colors, and considering independent black output IDs. Implement the independent entity ID option: permanent source pink ID, newly observed black output ID, and a process record linking input/output as an operation result rather than visual physical sameness. Initial inventory and pending-unpack queue are distinct from current products and process completion. One process must finish its right-bin drop before selecting the next source.

- [x] Assets/physics: retain the three supplied originals and reproducible normalized assets under `resources/cad`; use millimeter units, actual bounds and body-centered meshes. Keep a hollow collision representation for the box. Explicit demo masses are not inferred from nonwatertight CAD.
- [x] Scene editor/API: `kind=cad`, asset IDs `blue_box`, `pink_wrapped`, `black_core`; catalog supplies labels, true meter dimensions and default colors. Add/remove any number of instances, named/hex colors, position/yaw. CAD dimensions stay fixed; tables remain editable primitives. Preserve historical replay compatibility.
- [ ] Planning: 10mm clearance throughout, including existing saved setting migration and attached placement height. Verify both gripper Base sets have 12 corresponding spheres; do not change robot geometry or regenerate unchanged workspace atlases unnecessarily. Update measured bin completion to the actual CAD dimensions without using simulator poses.
- [x] Lifecycle: freeze original pink IDs/numbers; create the output entity after current Decisions confirmation at dual observation; bind all black-stage geometry, grasp, attachment and drop to its new ID. Preserve an explicit process record and count process completions once. Replace old transport-list wording in UI.
- [ ] Validation: focused regressions, full feasible suites/build, local/Ubuntu source parity, CAD scene visual/physical inspection, and an actual two-process run with archived images, candidates and video. Previous primitive-scene runs do not satisfy this revision.


2026-10-21 用户观察顺序修订：粉件放置、VLA拆包都在动作完成的当前位置做结果Decisions；拆包成功才回观测位姿，用新帧做抓取前Decisions和三RGBD建图。两段抓取在plan_grasp入口复核实际双臂观测姿态及当前地图同帧绑定。结果帧verification_capture及私有RGB支持后续跨帧关联，禁止跳过结果核验先返回。普通拆包采图预算随之为1+5N。黑件独立ID是工序记录；自由落体后仅维护右盒集合占据，不以落点重派ID；完成仍需本次执行链和新增可见数量确认，遮挡不确定停止。
按机器人前方+X、左侧+Y核对，两CAD盒及小桌中心X=.315m，盒朝机器人侧外边与主桌近边同为X=.16m；左右盒Y仍±.605m，默认两粉XY=(.315,.525)/(.315,.685)，中心间距16cm，完整CAD投影距底部真实内腔轮廓至少74.687mm、相互AABB净距97.213mm。固定用户关节和外参不变，单核离线1–4件渲染均可见且初始接触为零，默认camera2有效像素2945/3598；仅为配置/固定姿态QA，不代表抓放可达或新布局动态稳定性。

2026-10-21：右盒普通附物 plan_pose 使用实测盒区中心作为 TCP XY，保留实际 lift 姿态；完整持物点云加测量余量的水平投影须在盒区内，不强制物体参考点精确对中。10mm 三维碰撞检查、高度来源和五次插值/时间延长保持不变。
