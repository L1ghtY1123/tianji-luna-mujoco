# Visual inventory and controlled release

Approved scope: preserve task membership and fixed IDs; use masks, RGB and measured depth together; discover unmentioned obstacles from RGBD; reuse the measured table cuboid and map masked entities; release 10 mm above measured support, never above the rim as a fallback. JEV chooses skills and recovery.

Implementation ledger:
- Baseline: f07c689, clean Windows and last verified Ubuntu checkout.
- Ruling: continue in the user-named checkout for this already-authorized update and deployment; do not duplicate large resources or build images.
- Ruling: preserve existing public skill entries. New measurements are internal algorithms and evidence, not new autonomous recovery branches.
- Ruling: keep release-only task acceptance, and record visual destination confirmation separately. Visual unknown does not silently undo completed releases.
- Release implemented: fixed requested support-relative gap, no rim/attachment/voxel auto-raise; actual TCP checked within 2 mm and 2 degrees before opening. Rigid hold without slip remains an explicit assumption.
- Observation implemented: joint mask/RGB/depth evidence and exact source/destination ID-set deltas; peer departures or missing IDs return unknown. Accepted transfers no longer contaminate later comparisons. Full scene observations preserve obstacle IDs independently of task membership.
- Inventory implemented: bounded existing cube/sphere/cylinder/object queries discover environmental objects, including unmentioned objects; generic object alone missed workpieces in a real SAM3 probe. JEV membership remains the gate to task targets.
- Mapping implemented: table reuse and proven-empty background reuse; per-capture instance grids are still fresh. Fixed the wrapper reset that initially defeated the empty-grid cache. Raw background integration, explicit reset and failures invalidate reuse.
- Validation and skill documentation complete; synchronize this commit to Ubuntu and reload the existing GPU service. Web source uses its existing automatic reload mechanism.

Known evidence before changes: the existing region fixture requests 0.01 m but generates a 0.105 m object-floor gap because the candidate generator raises to the measured rim plus collision allowances. Existing tests explicitly asserted that obsolete behavior.

Runtime evidence on existing Ubuntu images:
- GPU measured-table smoke passes through the production mapper wrapper: reused table, zero semantic background surface, same empty grid on repeat, fresh selected/container layers, below-table samples removed, explicit reset invalidates table.
- Actual SAM3 on rendered three-camera RGBD passes initial/fine detection and temporal association for a red task object, blue destination and unmentioned black obstacle. All reconstructed points have measured pixel provenance; independent fixture truth is used only by the test scorer. A box called cube by one wrist is unified only after bilateral mask projection and depth checks (80% projection overlap, at least 30 points each way, shared measured surfaces within 5 mm with at least 30 samples and 20% coverage each way, no forward depth conflict).
- Recorded actual held RGBD replay through an isolated candidate GPU service rejects all 18 yaw candidates at 10 mm. The 16-sphere measured attachment extends 13.97 mm below the measured source support, so its conservative envelope conflicts with the requested floor gap. No goal was raised, gripper command issued, or collision threshold relaxed. This is **not a successful low-placement motion acceptance**. The remaining issue is the attachment/floor collision approximation; it must not be hidden by a higher drop.
- Mask IoU 0.75, centroid 3 pixels, RGB channel difference 30/255 after 3×3 smoothing, depth 10 mm and changed-area thresholds are initial engineering values, not measured accuracy. Real-camera noise/lighting calibration and all three full motion routes remain unverified in this update.

Final automated checks: Windows **1,448 passed / 53 skipped**; frontend **246 passed**; Vite build passed (existing >500 kB bundle warning); Ruff and diff checks passed; all 25 registered skill entries have handbook sections. Independent review findings were reproduced and corrected, including rejecting cross-class aliases explained only by mutual occlusion. No SDK/vendor source, robot model, image or dependency was changed.

Ubuntu evidence is archived in `data/visual-inventory-release-20261015/`: `mapping-gpu.log`, `fixed-release-gpu.log`, `perception-live.log` and `perception/`. The isolated test GPU process on port 18310 was stopped after validation. The 96×96 mapping fixture's repeated table lookup was 0.14 ms and full repeated build 21.9 ms; these are warm synthetic-probe timings, **not** an end-to-end speedup claim.
