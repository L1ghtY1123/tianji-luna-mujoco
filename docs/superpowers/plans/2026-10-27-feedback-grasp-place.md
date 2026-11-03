# Feedback-driven grasp/place implementation plan

> Use superpowers:executing-plans. The user approved the preceding design on 2026-10-27; execute continuously within this project.

**Goal:** Calibrated, configurable grasp candidates; phase-specific contact; native first lift before verified attachment; feedback-driven execution without mandatory one-second pauses.

**Architecture:** Keep Responses graph ownership and Decisions perception unchanged. Implement physical behavior in existing simulation, runner and project-side public SDK adapters. Advanced settings are generated from the validated schema and archived in each run.

**Tech stack:** Python/Pydantic, MuJoCo, pinned cuRobo 8e734f3ced1df898990bcd92de40abce475907db, existing JavaScript UI.

**Spec:** User-approved design in this conversation: ordinary-obstacle clearance is configurable, intentional pad/target contact is narrowly scoped; attach after the native first lift and current holding verification; conditions and timeouts replace mandatory dwell; no task nodes inserted by code.

## Global constraints

- Work only in the existing Windows/Ubuntu project; preserve unrelated dirty work. No branch reset, new parallel checkout, SDK/vendor modification, or unrelated service changes.
- New user approval supersedes former 10mm minimum, mandatory insertion/direction filters; the latest user instruction restores attachment only after first lift. Preserve ordinary collision checks and honest evidence scope.
- Environment geometry remains measured TSDF/ESDF plus table cuboid; no simulator object truth in perception/planning.
- Keep model-call budgets, task graph ownership, identity thresholds, runtime failure state, CPU limits and generation checks.
- Use public SDK interfaces; remove the superseded early-attachment support exception. Do not globally disable obstacles or allow object penetration to make a plan pass.

## Tasks

### 1. Settings and candidate geometry
- [x] Add failing tests for sub-centimeter noncontact clearance, independent release gap, adjustable insertion/direction constraints and zero default extra dwell.
- [x] Expose configuration through PlanningSettings and existing schema UI; remove forced saved-setting upgrades. Pass candidate settings explicitly and keep original poses.
- [x] Preserve native wrapper constraints through project planning_contracts, not vendor edits. Verify calibration mapping using own robot geometry and existing fixtures.
- [x] Run affected configuration/candidate/UI tests.

### 2. Feedback completion
- [x] Add failing tests for transiently slow gripper, stable feedback, timeout, reset/generation, and trajectory endpoint velocity/position.
- [x] Implement configurable stability windows and timeouts in Simulation; pass execution settings through runner. Extra dwell remains optional, default zero. Timeout is failure, never success.
- [x] Release waits use robot feedback and explicit configured settling allowance; do not read hidden object truth to declare visual success.
- [x] Run simulation/execution tests and update skill documentation.

### 3. Native first lift, then verified attachment
- [x] Latest user request: test detached native lift, then current holding verification and attachment. Collision, motion failure or lost grip prevents attachment and preserves provisional state; no task graph changes. Four regression cases failed before the change and passed after it.
- [x] Reuse native first lift after actual closed-robot validation, without attachment or support exemptions. Verify current holding after lift, then attach measured surfaces for subsequent carrying. Do not claim the first lift checked carried-object sweep.
- [x] Keep pad-only target exception and full other-obstacle checks. Result metadata records first lift as detached; documentation and skill contracts match.
- [x] Run contact/carrying/runner regression tests, including loss of grip and partial side effects.

### 4. Deployment and physical validation
- [x] Run affected suites, full feasible suite, lint/UI tests and SDK integrity check.
- [x] Deploy only owned services with backup and current-state check; no active experiment interruption.
- [x] Run actual Responses/Decisions MuJoCo tasks repeatedly, inspect every grasp/lift/release and timing. Iterate based on archived failures, not relaxed fact checks.
- [x] Fresh review of changed files, update validation record with exact results and remaining limitations.

## Review focus

1. A moving gripper at timeout cannot be counted as stable holding.
2. New parameters must reach solver/execution, not only UI metadata.
3. Detached first lift must retain actual robot/environment checks and truthful missing carried-sweep scope.
4. Feedback/collision failure after closure must retain provisional/held state and prevent blind replay.
5. Removing direction filters must not bypass geometry validation or reapply insertion.

## Execution ledger

- Started: read project boundaries and pinned public plan_grasp source. Existing SDK has link-based contact disabling, not object-pair allowances. Current project already has pair-specific execution checking and a measured-support departure path.
- Ruling: use this project in place, no per-task commits because the baseline contains extensive prior uncommitted work; review this turn using saved baseline copies and explicit changed-file list.
- Ruling: user approval authorizes updating old constraint documentation; no additional confirmation is needed.

- Settings/candidate/feedback/attachment targeted RED→GREEN: 18 tests passed; broad regression 91 passed with old-policy assertions requiring update.
- Ruling: first validate the original native lift against measured closed fingers + carried surface; do not solve the identical motion again if that command passes. Failure retains holding and stops. Support contact still needs physical validation.
- Deployment 05:23:01 UTC: 21 project files; backup before-feedback-grasp-20261027T052301.tar.gz. SDK untouched.
- Experiment 83 (476ceb8098474077b48336a7d9bacf8b) cancelled by an external /api/reset at 54.9s during observation; no grasp result. Do not count as success/failure of grasp.

- Experiment84: 97.0s, stopped before lift on attached sphere/source floor collision; grasp plan8.01s, held feedback retained. Experiment85 confirmed zero-clearance sphere penetration at initial support, not merely margin.
- Ruling: partition only registered measured floor returns into a separate TSDF/ESDF layer, retaining all nonfloor/wall returns. Permit carried-sphere support overlap only while raw measured surface stays above/inside support and departs with bounded normal motion; all other geometry checks remain. Cost if wrong: conservative false rejection or incorrect support attribution; independent review and physical tests required.
- Targeted support/contact regression76passed; browser354passed. Full suite rerunning.
- Independent release gap now persists and reaches generic and explicit-region goals; actual minimum remains max(gap, hard clearance).

- Independent review (gpt-6-astra) found two P1 issues, both fixed: removed nonsupport lift clearance recovery, and corrected independent release endpoint upper bound (RED→GREEN regression). No additional concrete bypass reported.
- Actual GPU ToolPose probe: both128-sample link poses convert successfully using public get_link_pose(make_contiguous=True); no SDK edit.

### Latest user correction: lift before attachment
Removed the measured-support partition/exception implementation. The first native lift checks the actual closed robot without carried-object spheres. After successful motion, fresh holding feedback gates attachment. This explicitly does not certify the first lift object swept volume. Physical validation follows; previous experiments 84–90 did not establish completion.

- Post-correction local validation: targeted62passed; complete3316passed/58skipped in346.63s; new four-case postlift test passes. Frontend354passed and production build passed before this physical-order-only change.
- Deployment06:16:28UTC:30files, remote measured_support.py retired, semantic_builder restored to pre-turn baseline. Backup before-feedback-grasp-20261027T061628.tar.gz.
- Experiment91 (86af6c69b76e479195207aed37ef5087): pink grasp/lift/attachment/place and VLA succeeded; native lift1.70s, measured held verdict then attachment. Full task stopped at black sourceIK0 in195.85s, Responses3 (initial graph required two repairs). Keep this failure; do not count as full completion. First256 black poses contained no downward tilt below45degrees, observed cloud724points from camera3 only. Counterfactual diagnostic IK at+10cm yielded21/100 solutions; original0/100. This does not itself identify collision versus reach as the unique cause, and no counterfactual command was executed.

### User correction: both VLA arms, reachable right output region, 5mm
The user explicitly requires both calibrated VLA start joint configurations, output on the right tabletop where B can grasp easily, and 5mm hard clearance. Restore A/B goals from existing project calibration; retire idle_arm_pose choice. Store capability configuration in resources/vla_unpack.json, sample only the configured right region and retain physical tabletop/workspace checks. Publish the region prior to Responses, never the sampled physical identity/pose. Update default and persisted active clearance to5mm. Preserve postlift attachment, task graph ownership and measured collision maps. RED:8 targeted failures; GREEN:32 passed after implementation. Physical reach/grasp validation remains required.

- Experiment94: the first dual-preparation/5mm task reached both calibrated arm joint goals (4.1s simultaneous execution); pink pick/place and VLA effect completed. Full task stopped at visual product verification (black probability0.05, wrapper removal0.92). Physical output XY[.59455,-.24038] was hidden under the right forearm; it was inside the then-configured region but not a suitable output location.
- Offline scene-authoring visibility check:36 right-table samples identified the forearm occlusion. Refined the capability output region to centerXY[.48,-.18]m, size[.04,.04]m; nine center/corner/midpoint samples all expose the part to right-wrist camera1 (2587–28548 pixels at640x360). This uses CAD only in a separate MuJoCo authoring model, not perception or planning. It is not an IK or physical grasp proof.
- The5mm regression exposed an independent-release-height bug: nominal release reach screening used hard clearance as height. Added a failing regression and changed the context to carry the configured release height; nominal height now uses max(release height,hard clearance), matching placement semantics. Affected121tests passed. Static review found no concrete defect in this change or the refined region. Full suite and actual experiment95 are running.
- Deployment07:03:05UTC:42files, backup before-feedback-grasp-20261027T070305.tar.gz; persisted live hard clearance confirmed0.005m. No SDK or other service changes.

- Experiment95 stopped before grasp because Decisions returned HTTP503 while selecting mapping scope; full task not completed. Experiment96 was interrupted by external source changes to src/tianji_sam6d triggering the web server hot reloader; it had reached mapping but no valid grasp result. The owned web service was restored after shutdown stalled; unrelated source files were preserved and excluded from this turn's deployment manifest.
- Refinement [.48,-.18] was rejected by the full regression: the conservative padded object envelope overlapped a right finger sphere. Final region is centerXY[.60,-.16]m, size[.04,.04]m. Nine actual rendered sample points have at least one visible camera, including the left-wrist witness where the right wrist is occluded. The full region envelope, every flat yaw and2cm geometry padding clear the dual VLA robot by at least28.24mm at sampled points; the analytical region-envelope test also passes the configured5mm threshold. Current27physical/config tests plus new nine-point camera-union test passed. This remains authoring validation, not grasp success.
- Deployment07:16:20UTC:41reviewed files, backup before-feedback-grasp-20261027T071620.tar.gz; excludes unrelated SAM6D client work. Full regression7 and experiment97 follow.

- Experiment97(e33a0d371b574d66bf28e6604060278c): pink grasp/lift/place, both VLA arms and visual product verification succeeded; B obtained a grasp trajectory, then stopped before lift on raw gripper angle-0.061318rad outside[0,0.8]. This is not a completed black grasp. Reconstructed physical contacts showed one terminal pad intersecting the moved black body; current wrist views were clipped at image boundaries. Pixel count alone was insufficient authoring evidence.
- Tightened authoring validation without using CAD in runtime perception: final output region centerXY[.44,-.30]m,4cm square. All nine points have full head projection in frame and visible rendered target; the full region envelope with2cm padding retains5mm robot clearance. This is farther right on the tabletop.
- Reproduced a separate feedback bug with4failing tests: stable out-of-limit positions were classified as held/empty. Stable closure now validates raw authored joint limits before returning; position verification returns unknown for an out-of-limit stable window. Retains the existing0.0001rad numerical noise allowance. GripperLimitError carries a specific error code and actual/min/max/command values; no clamping, motion insertion or limit relaxation. Affected50tests pass. The preceding full suite7 passed3320tests/58skips; suite8 covers this new guard.
- Deployment07:33:23UTC:42files; experiment98 and full suite8 running. SDK unchanged; no other task files deployed.

- Final configuration: centerXY[.44,-.30]m,4cm square, both fixed VLA arms, live hard clearance5mm. Full suite8:3324passed/58skipped/one dependency deprecation warning in389.49s; reviewed source/test files Ruff clean. Independent review found no new limit-guard defect.
- Experiment98(bec3e4d4764c4cedab9d99f9c5a30001): full physical task succeeded in247.17s, Responses2. Both pink/black grasps lifted, then verified holding, then attached. Black physically inside right box, ledger lineage valid, both arms returned to task start with ~0.000004degree maximum error.
- Experiment99(d7e1b825e18d45ef829235aaae32a21e): unchanged code, new random output, full task succeeded in319.12s, Responses2. Independent physical containment, lineage and dual return again passed. Initial measured black centroids differ between runs:98[.43074,-.29054,.61041],99[.43078,-.28359,.61054]. These are measurements, not simulator pose priors.
- Scope limits: strict single-Responses audit remains false for both because call count was2; do not describe that metric as passing. Experiment98 plan_grasp calls A11.21s/B9.69s, not1–2s. Two final-configuration successes do not prove arbitrary-task reliability. Detailed deliverable: docs/vla-dual-arm-validation.md and data/vla-dual-arm-validation.json.
