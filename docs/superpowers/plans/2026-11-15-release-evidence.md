# Release height, delivery semantics and Decisions input

User-approved corrections: compute TCP Z from measured destination support + object height + margin; allow an additional 10 mm when the conservative attachment needs it. Keep fixed XY, grasp rotation, same arm and Luna. Arrival at measured release XYZ within tolerance followed by release supports inferred delivery. Container delivery means entry into the named container; the configured release square is not a final-resting-position requirement.

Implementation is inline in the existing repo. No agents, worktrees, SDK edits, deleted collision checks or new product test directories.

- [x] Freeze source and real failed inputs. Compare image, state and question changes individually, then a combined variant. Keep regressions in the report and adopt only supported changes.
- [x] Use held CAD/TSDF geometry to measure vertical object height independently of collision balls. Preserve actual grasp offset to compute physical bottom evidence and prevent an above-object TCP from underestimating clearance. Increase configured release margin to 20 mm after reproducing remaining conservative-envelope conflicts.
- [x] Separate the fixed release region from destination acceptance: container entry is success, while table-region tasks retain their requested area. Inferred success is backed by actual arrival, target holding and release evidence; visible outside-destination evidence still contradicts success.
- [x] Deploy the supported current-contact holding view and evaluated identity input changes; keep bounded multiview correction and unknown/no_match paths.
- [x] Run focused offline checks, static checks/build and serial complete simulations. Reclassify the prior entered-container case under the corrected user contract, without rewriting its original sensor evidence.

Alternatives: shrinking/deleting attachment balls would compromise observed geometry coverage; retain coverage and use the explicitly allowed extra margin. Universal prompt shortening and replacing three inputs together are experimental alternatives, not assumed improvements. The first frozen holding ablation already shows regressions from indiscriminate shortening.

Results: four primary full simulations physically delivered 4/4 and program success was 3/4. A predeclared final-prompt simulation also delivered and passed, for 5/5 physical delivery and 4/5 program success overall. Preserve the cluttered-container false rejection as unresolved evidence; do not claim stable reliability. Detailed results: `docs/release-evidence-20261115.md`.
