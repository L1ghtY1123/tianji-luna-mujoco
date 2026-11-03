# Placement selection and graph reliability

Responses, Decisions, the inventory and permanent IDs remain the policy foundation. Responses chooses
each object's immutable requirement and independent subgraph once. Runtime Decisions owns unexpected
recovery nodes and edges. Skill code does not add a task route or choose a different destination.

## Evidence leading to this change

Baseline run `45f483a2ea87432dba7a96c1665e9977` failed before motion: initial Responses produced two
subgraphs with actions but no `end` nodes or return edges. The previous generic diagnostic was
`graph_no_possible_completion`. This is a graph-authoring failure, not a SAM3 or grasp failure.
The prompt and EndNode schema now explicitly explain scope returns. Validation identifies the scope
and its dangling leaves; it does not repair model topology. Runtime Responses remains disabled.

## Placement responsibilities

- Responses chooses a destination matching the original language and explains it in the object's
  `task.reason`. General tabletop placement can use the measured table. A configured processing input
  square is selected when appropriate to the request, not as a universal destination.
- Decisions confirms destination semantics against the original task independently of the proposed
  per-object task. This is an extra question in the existing binding call, not an extra Responses call.
- `identify_destination.placement_preference` selects `clearance` or `reach`. The first prefers measured
  obstacle separation up to the configured soft activation distance, then arm reach; the second favors
  reach. Both retain every numerical safety check. Containers keep their measured opening/drop solver.
- `robot_skills/placement_selection.py` owns numerical ranking and exact goal-workspace validation.
  Measured obstacle XY bounds rank candidates only. They are not added to the collision map and do not
  certify unobserved space. Candidate generation insets the carried footprint inside the chosen region.
- `plan_place` retains up to 12 spatially distributed candidates. After height resolution it checks the
  exact final gripper-front XYZ against the calibrated workspace, then IK, held-body and TSDF/ESDF path
  checks. Precise configured XY never moves. Candidate exhaustion returns evidence to Decisions.
- The result records the preference, observed obstacle IDs, missing geometry, candidate positions/gaps,
  exact workspace result and rejected candidate reasons. Occupancy ranking is explicitly not a collision proof.

GaP ideas adopted here are explicit scope returns, outcome edges and independently described capability
contracts. This does not make our repository identical to GaP: shared controller/Runner context and legacy
`demo_*` implementation modules still need gradual extraction. Moving code without validated boundaries
would not establish higher physical reliability.

## Validation

See `data/placement-policy-tests.txt`, `data/placement-policy-all-tests.txt` and the experiments appended
below. Report success rates only for repeated runs with the same code/configuration; the baseline does
not belong to the modified-version success count.

Run `9e64947093744f2e865a622fdf6fa906` completed the two-object task in 295.72 s, with one Responses,
25 Decisions and no recovery patch. Both real grasps/placements, per-object return-and-observe and
final measured initial posture were verified. Pink TCP XY was (0.39807, 0.12587) m inside the configured
square; green TCP XY was (0.32200, 0.22838) m on the measured table. Both first candidate paths succeeded.

Run `8f99806297e743c5ac0a57630303df62` moved both objects successfully but the overall task failed.
Its model graph correctly observed after green delivery, then returned to initial pose inside the green
subgraph. The runtime incorrectly demanded camera freshness at subgraph exit to prove that historical
observation. This was a runtime evidence-lifetime bug, not missing Responses orchestration.

The correction records the receipt immediately after a successful inventory-observation skill applies
its effects. It is bound to simulation generation and the specific verified delivery, survives later
arm motion, and cannot prove another delivery. Missing completion evidence now produces unknown at
the owned subgraph's success end. Decisions can choose registered recovery skills and recheck only
that end. Earlier physical nodes are never replayed; runtime does not choose a recovery sequence.
Regression coverage explicitly exercises observation-before-return and end recovery without regrasping.

## Final deployed revision and physical checks

Code/contract digest: `ee4ac711216a5d2dda82713f9762bdcc37484be7f8349af0c2eeefc557582f1c`.
The deployment manifest verified 94 files with no differences. Final full backend verification:
**3524 passed, 58 skipped**, plus targeted regression coverage in `data/placement-final-regression.txt`.

| Run | Instruction order | Whole-task result | Responses / Decisions | Runtime patches | Duration |
|---|---|---|---|---|---|
| `a750730c024944bba9e8590df647474f` | Pink to configured square, then green to measured table | Succeeded | 1 / 29 | 3 | 334.36 s |
| `5d7b9847f11a4d98b80f69d37f52f8a5` | Green to measured table, then pink to configured square | Succeeded | 1 / 24 | 0 | 473.30 s |

Both runs performed the real MuJoCo grasps and placements, returned both arms to the observation posture
after each delivery, completed the corresponding inventory observations and verified the final measured
initial posture. Both preserved independent per-ID subgraphs. In the first run, current green geometry
was missing at `identify_target`. Decisions selected `observe_scene(capture_mode=fresh)`, then retried
only target binding with the same permanent ID. The three patches are entry into the recovery router,
the selected observation node and edges, and the selected return edge. This was a fresh capture at the
current posture, **not** a demonstration of viewpoint-changing recovery. No runtime Responses was used.

The final Decisions RGB inputs and measured joint/observation receipts are saved as
`data/native-final-<run-id>-evidence.json`. Compact placement, workspace and policy evidence is in
`data/placement-audit-<run-id>.json`. Earlier failed runs remain failed; they are not counted as final
revision successes. These two successful examples are not a statistical success-rate estimate or proof
of arbitrary grasp-place/VLA tasks. Video encoding runs after robot completion and may report sampling
gaps separately; robot status and video status must not be conflated.

An additional post-run evaluation of the reverse-order scene confirmed that conservative full-object
bounds for both items lie inside the table XY bounds. Minimum margins were 88.9 mm for pink and 6.1 mm
for green. This explains the green object's near-edge appearance; the current preference is not an
optimizer of table-edge margin. This evaluation used simulator state only after execution and was
never supplied to Responses, Decisions, perception or planning. See `data/placement-run29-postcheck.json`.
