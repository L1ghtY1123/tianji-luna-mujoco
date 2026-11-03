# OBB grasp and scene-query placement implementation

Approved specification: the five-point design accepted in this chat on 2026-10-30.
Use observed object geometry to derive an OBB, sample its mesh for GraspGen, and
create a covering attachment only after verified post-lift holding. Preserve grasp
rotation. Search XY on a measured surface/container region, solve bounded lowest
clear Z against the existing world, and reuse the calibrated TCP workspace atlas.
Decisions judges original-language suitability of candidate IDs outside geometry
skills; the program binds IDs to poses. All placement uses plan_place.

## Execution

1. Remove installed and repository JEV skill copies and repair their links.
2. Add observed OBB geometry and GraspGen mesh sampling with frame/provenance tests.
3. Bind the same OBB to actual grasp TCP and fit bounded covering attachment spheres;
   retain original measured points, post-lift lifecycle and scene-map geometry.
4. Replace XY envelope rejection with existing-world sphere collision queries;
   preserve support footprints, precise XY, lowest-Z bounds, workspace and three-phase planning.
5. Verify Decisions unknown/rejection and unchanged original-language/ID bindings;
   update contracts, prompts, README and skill guide.
6. Run focused tests, affected tests, full feasible tests and lint; perform one
   independent review of this turn's diff and fix substantive findings.

## Boundaries and review focus

No vendor/SDK modifications, no CAD or hidden scene truth, no runtime Responses,
no hidden observation/motion, no changes to permanent identity or completion policy.
Review frames, OBB fitting provenance, attachment containment, batch correspondence,
stale scenes/settings, occupied XY, containers, unsupported/no candidate, and floating
release rejection. Local unit tests are not evidence of live robot success.

## Workspace and progress

Use the existing dirty checkout per AGENTS.md; no worktree or unrelated commit.
Baseline: data/obb-placement-20261030/baseline.zip (including installed JEV skill).
Ledger: data/obb-placement-20261030/progress.md.

## Completed verification

- Python full suite: `python -m pytest -q`, 3595 passed, 58 skipped, 404.45s.
- Web: bundled Node `--test --test-concurrency=1 tests/*.test.js`, 380 passed.
- Ruff on this turn's modified Python files: passed.
- One independent review: no substantive source defects; no live GPU or physical claim.
- Installed and repository JEV skill paths are absent; originals retained in baseline.zip.
- Follow-up deployment completed after explicit user authorization: existing SSH identity
  connected successfully; earlier unified-placement dependencies were included. Ubuntu
  focused regression: 119 passed. Web/cuRobo healthy after restart, remote Decisions
  credential configured, layout/settings restored and file hashes verified. Backups and
  logs: data/obb-sync-20261030/. Full live grasp/place and GPU query performance remain unverified.
