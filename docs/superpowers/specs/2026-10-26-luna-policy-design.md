# Luna object-grounded policy architecture

User authorization: 2026-10-26, implement the agreed design in tianji-luna-mujoco, restructure owned code and remove obsolete code. No JEV runtime or JEV skill dependency. This specification incorporates the user's corrections; it is an implementation record, not another approval gate.

## Required behavior

Responses understands natural language, proposes visible entities and generates a bounded policy graph once per normal task. Decisions supplies visual and semantic judgments during execution. Deterministic code validates geometry, identities, versions, budgets and effects. Robot skills expose explicit contracts and registered handlers. A new skill must not require extending a hardcoded policy Literal or editing an interpreter dispatch chain.

Task inputs include object selectors/IDs, source, destination, quantity, processing requirements, constraints and completion conditions. No color, table-only destination, robot-side or unpack route is mandatory in the generic entry. Permanent IDs persist, appearances and observations are versioned, process outputs get separate IDs with provenance. Completion checks obligations plus independent current semantic evidence; an empty queue alone never proves all targets were found.

Initial Responses call target: one, independent of target count. Up to two additional attempts cover invalid generation and novel-event revisions; no repeated revision on unchanged evidence. Common recovery is represented in the generated graph, never silently selected inside a skill. Revisions preserve original goals, constraints, completed effects and remaining obligations and are applied only between completed actions.

## Boundaries

- agent/: typed schema, Responses compilation/revision, Decisions judgment transport. No simulation or motion execution.
- runtime/: policy interpreter, inventory/obligations, versioned state, execution lifecycle and composition.
- robot_skills/: bounded action handlers and configured VLA simulation adapter. No Responses orchestration.
- Existing sensing, geometry and planner adapters retain validated public SDK use and remain reusable; move cohesive groups only with import/test migration.
- skills catalogue: extensible contracts, descriptions, argument schemas; no JEV requirement.

VLA unpack is one bounded skill. MuJoCo removes the selected pink wrapper and creates the black product at a configured fixed pose. Motion to the required station is explicit; visual verification is a separate Decisions step. Simulation truth never establishes perception, identity or task completion.

## Mapping safety

User correction: collision mapping contains only observed inventory entities; filtering other objects is intentional. Three cameras do not by themselves certify complete coverage. A partial observation must not erase previously measured inventory geometry. Reobserve unchanged entities incrementally; explicitly invalidate moved/processed entities and retire consumed ones only with verified lifecycle evidence. Keep original capture IDs and provenance for retained surfaces.

Raw calibrated RGBD may be retained separately to establish visibility/free-ray evidence; it does not add unlisted objects to the collision map. Unknown, observed free and occupied space remain distinct. A motion requires coverage for robot links and carried geometry along its swept path, not just endpoint/TCP/target coverage. Invalid depth, occlusion, outside-FOV and stale geometry cannot be treated as free. Existing trusted static geometry may be retained with explicit provenance; no hidden MuJoCo ground truth may supply maps.

Insufficient coverage returns observation_coverage_insufficient with evidence. Only an independently validated observation motion can gather more views. If no such motion exists, stop and report the missing coverage; do not bypass the guard. Physical continuous-path guarantees and actual capability limitations must be documented, not overstated.

## Validation

Behavioral unit and integration tests: graph arguments and branches, retries and budgets, unchanged-event deduplication, immutable task on revision, persistent IDs/counts, general sources/destinations, truthful completion, VLA fixed output and separated verification, inventory-only incremental mapping and unknown-space guards. Real MuJoCo and available GPU service checks supplement CPU tests. All changes stay in the user's original authorized directories; no SDK edits, no extra worktrees, no service changes to other projects. CPU limits remain in force.
