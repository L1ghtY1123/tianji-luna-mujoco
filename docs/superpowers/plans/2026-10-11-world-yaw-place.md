# World-frame yaw placement implementation plan

> Execute the approved design in this session; independent frontend and fitting work use the dispatching-parallel-agents workflow. Keep verification evidence in data/place-yaw-20261011.

**Goal:** Tight cached attachment spheres (at most 16), fixed world-frame object XYZ with eight world-Z yaw candidates, complete collision-checked place phases, and an obvious point-cloud hide control.

**Architecture:** Keep the existing skills and JEV selection. Pure placement geometry consumes the exact uploaded spheres and measured grasp/table/bin references. A project-side GPU batch endpoint uses the pinned public cuRobo APIs for attached approach/place; the application checks commands and open/detached retreat before choosing a complete candidate. SDK and vendor source stay unchanged.

**Spec:** User-approved discussion immediately before “确认,执行”: h=max(requested height, sphere bottom overhang + hard clearance); eight yaw candidates, original roll/pitch and fixed object reference XYZ; TCP translation follows the rotated grasp offset. No XY/Z search or arbitrary tilting.

**Constraints:** cuRobo commit 8e734f3ced1df898990bcd92de40abce475907db; initial three-camera visible cloud only; same collision scene and original robot spheres; no added RGBD noise; no automatic skill sequencing; sphere budget <=16; no new dependency or image. Repository has no Git, so retain dated file backups.

**Review focus:** World versus tool rotation with tilted/off-center grasp; exact uploaded radii without double inflation; first candidate retreat fails but later candidate succeeds; service errors are not hidden as infeasibility; hiding overlay during in-flight refresh stays hidden.

- [x] Tight-fit regression, bounded fitting and cache, cloud coverage and timing benchmark (attachment worker).
- [x] Regression for off-center world-yaw candidates and shared envelope-derived release height; implement placement helpers (root).
- [x] Real batched attached IK and two-phase planner, copied derivatives, per-candidate collision evidence and unchanged logical hold; focused tests (backend worker).
- [x] Application candidate selection, command and open-gripper/retreat checks, cancellation/error propagation; focused tests (root).
- [x] Discoverable persistent overlay visibility control; frontend tests/build (frontend worker).
- [x] Update place contract/docs/setting description; run affected and full feasible tests/lint/build; inspect all diffs (root).
- [x] Back up remote state/source, synchronize owned files, reload only own services between tasks; GPU replay and physical RGBD grasp/place acceptance, retain results (root).

## Progress

- 2026-10-11: recorded current failed-held state and local file baselines. No live simulator reset yet. Design and execution already authorized by user.

- 2026-10-11: local full Python suite 897 passed / 56 skipped; frontend 181 passed and build succeeded. Attachment fixture 16 balls, total ball volume -5%; cold fit 2.42 ms and cached 0.0175 ms. Frontend hide control synchronized to Ubuntu with backend instance unchanged. Browser automation unavailable (IAB timeout); HTTP source and DOM/unit tests confirmed.
- Isolated GPU/physical smoke reached a successful grasp and 8-way place batch, but all candidates failed. Height formula yields 22.93 mm; actual ESDF still rejects boundary spheres. Controlled +5/+10/+15 mm replay distinguishes floor-boundary costs from yaw-dependent wall costs; batched IK has no solution even after floor costs clear. Diagnosis in progress before backend deployment.

- Final: one explicit map voxel allowance added after pinned ESDF source and real GPU evidence showed near-surface seed band; all yaw candidates still share one height. Full Python 898 passed / 56 skipped; frontend 181 passed, build and scoped Ruff clean.
- Isolated RGBD/JEV/SAM3/GraspGen/cuRobo physical run 430d531b74a745e7be68cb10e2ac271c succeeded with an explicit left-arm instruction and existing plan_max_attempts=2. Selected world yaw +45 degrees, release height 28.919 mm, 16 attachment spheres. Final RGBD verification confirmed; independent physical scorer confirmed all corners inside box XY and object resting on floor. Default automatic arm choice and attempts=1 are not covered by this success (recorded prior failures retained).
- Synchronized 26 owned files to Ubuntu, verified hashes, restarted only own web/cuRobo containers, removed isolated service with restart, restored confirmed editor scene layout and idle robot. No images or dependencies created. Evidence: data/place-yaw-20261011/verification-summary.json.
