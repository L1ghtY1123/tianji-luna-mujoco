# Frozen pregrasp quality regressions

The JSON/NPZ contain only original observation geometry and accepted source-IK
FK goals from 3ce and 5be. `data/create_pregrasp_fixture.py` records the exact
source event paths. No held feedback, simulation object pose, ground-truth
shape or later motion result participates in the prediction.

Each cloud has 1,955 points: the exact 1,361 measured prefix and 594 explicitly
labeled height-completion points, with original provenance/uncertainty. The
original candidates retain nominal/achieved matrices, joints, score, round,
source indices and solver evidence. The historical Link3-only boundary retained only 3ce 931 and 5be 826. The
current owner-hand assembly boundary retains 3ce and all five 5be goals; the
original 6 mm padding remains. Added ffb includes all seven raw source-IK
successes from rounds 1/2: six quality-only cases pass the new boundary and
972 still fails the original direction rule. Its exact source extraction is
`data/ffb-quality-failure/create_fixture.py`.

This is not a motion acceptance fixture: 5be 826's archived lift failed.
