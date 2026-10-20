# Explicit mapping validation, 2026-10-15

The project now supplies actual collision-distance and sphere-radius coverage to the unchanged public native instance mapper. It preserves native TSDF allocation, voxel phase, camera rays and masked depth. This is not the rejected fixed-domain mapper experiment. Vendor files, installed cuRobo sources, images, live service state and simulator state were not modified by these probes.

## Implemented boundaries

- Minimum instance ESDF padding is `max(robot sphere radius, supplied attachment radius bound) + max(applicable collision distances) + 2 × 0.01 m`. The production caller supplies a conservative attachment bound from the full observed world-AABB diagonal plus uncertainty and attachment margin. The maximum robot radius in the GPU fixture was 0.13234879 m; the actual optimizer distance was 0.03 m and hard clearance 0.01 m.
- Padding rounds outward to whole 1 cm cells. A 1 nm numerical rounding tolerance is added before rounding and subtracted afterward, so required coverage is never reduced. Diagnostics report the minimum, effective padding and rounding addition separately. The fixture minimum was 0.18234879 m and effective padding 0.189999999 m.
- Measured table top equals the fitted height; the former 5 mm height addition is removed. Downward completion is at least `max(0.30 m, hard clearance)`. Native fitting clearance is `max(0.02 m, hard clearance)`. XY footprint and robot-facing crop remain native.
- Table evidence is cached across captures. Reset, calibration/intrinsics/registration changes and settings changes invalidate it. Reuse also checks current robot-sphere clearance against the cached cuboid; a violation forces refitting and rejects a still-colliding result.
- A separate crop model places the lower bound at measured table height. The public semantic crop and public TSDF crop remove points and cells below it before ESDF computation. The actual collision cuboid retains its downward extent.
- Unclassified background is removed only when the experimental setting and verified full-observation flag are both true, the map is semantic, and it is not an observation-motion map. Semantic object/container layers and table remain. Initial observation maps and held background-mode captures retain valid background. The empty background ESDF is still allocated; this change does not eliminate that allocation.

## Executed GPU checks

The checks ran as disposable Python processes in the existing `jev-mujoco-curobo-guangyu` container, with changed project modules loaded from `/tmp/jev-bounded-overlay`. They did not call the running HTTP service. Pinned cuRobo source identity was verified as `8e734f3ced1df898990bcd92de40abce475907db`; PyTorch was `2.11.0+cu128`.

`tests/bounded_mapping_gpu_smoke.py` used unchanged native `build_instance_esdf` for old and bounded padding. Three synthetic registered-depth fixtures varied object size, X position and depth. Sorted exported TSDF surface centers were exactly equal for 84, 162 and 338 surface voxels respectively.

| Fixture | Legacy ESDF cells | Bounded ESDF cells | Reduction | Bounded near-field overestimate vs nearest TSDF surface |
|---|---:|---:|---:|---:|
| Default | 2,920,155 | 123,552 | 95.77% | 0.0582 mm |
| X +13 mm, depth 547 mm | 2,801,513 | 110,000 | 96.07% | 0.0582 mm |
| X −27 mm, depth 523 mm | 3,123,737 | 148,016 | 95.26% | 0.0582 mm |

Each fixture queried 29,791 common world-space spheres using public `RobotCollisionChecker` at 10 mm and 30 mm activation distances. The bounded fields had zero missed contacts against the nearest retained TSDF-surface oracle with a 1 mm decision tolerance, at both distances in all three fixtures. This is a measured-surface test; it does not certify unseen object faces or whole-trajectory safety.

**The old and bounded ESDF fields are not equivalent.** Direct unrounded padding reduction produced up to 14.14 mm differences despite unchanged TSDF surfaces. Read-only inspection identified native float32 scatter conversion of surface centers at ESDF cell boundaries as a contributor. Outward whole-cell padding aligns the ESDF with the retained TSDF centers in these fixtures. It changes ESDF sampling phase relative to the old odd-sized grids: interpolated overlap differences were 13.54–16.04 mm and old/new collision classifications differed on 393–651 queries, depending on fixture/distance. The legacy fields missed 33–75 oracle contacts per fixture/distance; bounded fields missed none. No claim of bitwise field equivalence or universal geometry equivalence is made.

`tests/measured_table_mapping_gpu_smoke.py` passed on the updated project implementation: the initial map retained eight background surface voxels; the explicitly enabled full-observation map retained zero background surface voxels and discarded 48 unclassified returns. Both semantic layers remained. Three measured source returns below the fitted table were removed; all exported instance surface centers were at or above measured support. The table object was reused across captures and invalidated by reset. Measured top was 0.699999928 m and downward extent 0.30 m.

The existing `tests/perception_mapping_gpu_smoke.py` also passed through the updated `/tmp` modules. It verified registered RGBD mapping, non-target and released-source layers, actual gripper feedback changes, and reuse of the mapper when gripper geometry changes. The measured maximum robot-sphere motion between feedback configurations was 38.109 mm.

Logs are saved under `data/placement-mapping-20261015/explicit-bounded/`. Reproduce in the existing container with its normal project `PYTHONPATH`, running the two new smoke scripts directly. They use `/opt/jev` only as the read-only project resource root and create independent mapping objects; no service reset or motion is required.

## Timing and verification limits

Representative saved-RGBD cold/warm profiling was **not run**. Synthetic timings are retained in the logs but include shape-specialized compilation and variable GPU scheduling; they are not an end-to-end speedup claim. The measured result is a 95.26–96.07% reduction of the tested per-instance ESDF cell counts. Background allocation, TSDF allocation and specialization remain.

Mapping-specific CPU checks passed after the implementation and builder-reuse test update. An earlier full-suite run reported 1,407 passed, 53 skipped and five failures during concurrent integration: the stale mapper-constructor test was corrected, while four request-decision fixtures were handed to the parent agent and corrected there. The parent owns the final integrated full-suite run and deployment validation.

Read-only compatibility sources: public `curobo/perception.py`, public `curobo/collision_checking.py`, matching `MapperCfg` coordinate/extent contract, `Mapper.compute_esdf`, the official `test_motion_planner_esdf.py`, and native `volumetric_mapper.py` / `mixed_esdf_scene.py`. Internal scatter-kernel source was inspected only to diagnose the observed difference; no private SDK API is called by the new production code.

## Integrated verification before deployment

- Windows full Python suite: **1,431 passed, 53 skipped** (93.56 s). The final error-conclusion text addition was then checked by its 19-test targeted suite. Ruff and `git diff --check` passed.
- Frontend: **246 tests passed**; Vite production build passed, retaining the existing large-chunk warning.
- Four live JEV request probes passed: an environment declaration with one requested box object; explicit left-arm two-red/three-green targets (total five); all tabletop objects; and unspecified-count box objects. These probes used the configured JEV endpoint, without simulator object metadata or an OpenAI vision call.
- Skill coverage and local document links passed: 25 registered entries, 24 visible to JEV. Contracts and implementation documentation were updated together. New guard failures have Chinese conclusions explaining what was refused and why.
- SDK/vendor, robot assets and collision-sphere files are unchanged in this implementation. No image was created.

Logs: `data/explicit-mapping-20261015/windows-pytest.log`, `data/explicit-mapping-frontend-tests.log`, and `data/explicit-mapping-20261015/request-scope-jev.json`. Ubuntu full-suite and deployment results will be recorded after testing the committed candidate.

## Ubuntu deployment and first motion check

The exact candidate `ecd8049fde3d92d50c43f4d702dbfd0d925daeb0` passed an isolated Ubuntu full suite: **1,474 passed, 10 skipped** in 56.89 s. The live checkout and dependencies were untouched during testing. After archiving the idle run, Ubuntu main was fast-forwarded. The existing cuRobo container was restarted; the web backend reloaded mounted source. All service health checks passed. No image was built or pulled.

An independent MuJoCo box-to-table run used real JEV/SAM3/GraspGen/cuRobo, default planning settings, one 4 cm cube and no OpenAI assistance. It **succeeded in 104.54 s**, with no skill errors or rejected calls. Independent physical scoring placed the cube center at [0.386715, -0.004287, 0.619999] m, within the yellow 20 cm square centered at [0.38, 0, 0.60] m. The physical oracle was used only after execution, not as controller input.

Measured skill times were: initial observation map 5.07 s; dual-arm observation movement 10.58 s; fine observation 1.46 s; selected semantic map 9.20 s; grasp planning 10.44 s; place planning 16.45 s. JEV calls totaled 14.39 s. Selected mapping reported 7.93 s native build time, including 6.77 s instance processing, 1.08 s depth projection and 0.25 ms cached-table lookup. It removed 14,261 unclassified returns; the cube/box ESDF grids had 115,248/240,448 cells. This is one post-change run, not an A/B speed comparison.

The first harness launch incorrectly overrode the OSMesa web container with `MUJOCO_GL=egl`, producing a Renderer AttributeError before motion. Running with the container's existing OSMesa environment fixed capture; production renderer configuration was not changed.

A subsequent two-identical-object request exposed low confidence in the new selection-order classification for “all cubes”. The question was narrowed to whether source-object identity constraints are present, separately from region/type/count. The original confidence threshold remains. Five live request cases then passed, including the failing sentence; 34 request/decision regression tests passed. This clarification is the only runtime code change after the full suite.

The corrected two-object request reached its first placement, then rejected all 18 yaws at both half-region centers. Exact uploaded-sphere replay localized this to an unrelated ESDF-cell allowance applied to the directly measured marker polygon. Splitting boundary clearance (`radius + configured hard clearance`) from occupancy clearance (`radius + hard clearance + grid size`) preserves collision settings. The independent replay accepts 11/18 yaws per position, with best measured edge gaps 17.99/17.97 mm. A regression first failed on the original code, then passed with the split; 49 related placement tests passed. Marker measurement uncertainty remains; this is not a physical edge guarantee.

## Final verification and remaining acceptance gap

Final Windows suite at `f3005fc`: **1,432 passed, 53 skipped** in 89.73 s. Ruff, diff checks and all 25 handbook-entry/link checks passed. Ubuntu's earlier complete run covers `ecd8049`; subsequent changes were the live-tested request clarification and the 49-test placement-boundary correction.

The new two-identical-object run did **not** establish end-to-end multi-object success. In `route-identical-two-boundary`, the first grasp closed empty; JEV selected recovery, then encountered target ambiguity and ultimately `jev_input_budget_exceeded` after 62.76 s. It never reached the corrected placement step. No fixed retry or threshold bypass was introduced. The successful single-object route and boundary-data replay must not be presented as successful two-object transport or live post-place identity validation. Lifecycle associations are covered by the automated fixtures, while a complete successful multi-object motion sequence remains unverified.

All CLI simulation processes ended. The web task was idle; the dedicated GPU service's test carrying/map state was cleared, with health reporting ready, `attached=false` and `scene_token=null`. Existing image tags are unchanged. Temporary probe code was removed; 193.8 MiB of isolated test source/dependencies/cache was reclaimed after preserving validation logs and manifests. Run evidence remains under `data/explicit-mapping-20261015/`.
