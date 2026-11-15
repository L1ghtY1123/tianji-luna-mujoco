# Placement height and identity correction implementation plan

User-approved scope: keep Luna for Responses and Decisions; compute release height from the measured source support and actual grasp TCP, and correct initial identity choices using clear annotations and bounded multiview review. Execute inline; AGENTS prohibits subagents and parallel worktrees.

The fixed XY and grasp rotation remain the placement contract. Console Z becomes object-bottom clearance (default 10 mm) above the measured table or container rim. Retire the old fixed TCP-height settings without interpreting old 110/100 mm values as bottom clearance. Missing source-support evidence remains a failure. Collision geometry validates the goal independently and cannot change it. Before grasping, workspace selection uses an explicitly labelled estimate; the measured release goal is checked again after grasping.

Identity keeps SAM3 masks, geometric conflicts, dated references and transactional permanent IDs. Compare annotation variants on frozen historical inputs before adopting one. After tentative selections produce cross-camera groups, perform one joint-view subject selection with the new correspondence evidence. It may replace the first selection. Regroup changed seeds once, then retain the existing competing-owner adjudication and bounded reselection. Never repeat an unchanged question until accepted; unknown and no_match remain valid.

- [x] Save baseline and reproduce the fixed-height and irreversible-first-choice failures using offline checks in data/runs/placement-identity-20261115.
- [x] Update demo_placement, planning_settings, demo_algorithms, transfer_reachability and placement_resolution; verify height translation, A/B archived inputs, setting migration, missing support and endpoint semantics.
- [x] Compare labels on five frozen cases: blue container, green contents, dated reference, reversed choices and missing mask. Adopt surface_label; a broader opposite-bin and temporal-identity suite remains unverified.
- [x] Update task_subjects, subject_assignment and both caller adapters. Verify correction, no_match, unknown and one review bound; retain existing owner conflict handling. Real Luna corrected an injected shared wrong seed before any memory commit.
- [x] Refresh contracts and evidence input version, pass static checks/build, synchronize with backup and hashes. Restart both web and cuRobo because both import PlanningSettings.
- [x] Run four serial complete simulations and archive all failures, return status, runtime and model usage. Strict delivery acceptance is 0/4; this is a completed experiment, not successful end-to-end acceptance.

Outcome: implementation and targeted checks complete; end-to-end reliability remains unresolved. See [full evidence and limitations](../../placement-identity-20261115.md). Unknown-object collision envelopes, grasp recovery, inferred placement false success and measured container/region mismatch remain separate blockers. No model upgrade or weakened collision guard was used.

No SDK changes, no removed collision checks, no simulator truth in planning, no new product test directories. Existing CPU limits remain. Geometry modelling conflicts discovered by the new goal are recorded separately from goal calculation.
