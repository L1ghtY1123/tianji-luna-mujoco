# Continuous transfer and JEV recovery

User approved execution on 2026-10-12 after two rounds of design. This specification records that approval and the final correction: recovery decisions MUST be made by JEV, never an error-to-action program.

## Requirements

- Preserve RGBD-only environment evidence. Simulation editor state is never task evidence.
- At grasp/place completion, expose success/failure/unknown evidence. JEV chooses recovery, reobservation, regrasp, place replanning, defer or stop. Code enforces contracts and bounded attempt budgets but does not choose a replacement action. Clear physical/execution faults interrupt motion; recovery still goes to JEV.
- Simple gripper-feedback interruption is optional only while carrying, never during intentional release. No continuous VLM or observation loop during motion.
- Explicit arm in the instruction binds that arm and skips arm selection inference; only that arm needs IK. Never switch arms without instruction authority.
- Natural-language source sets support single/all/count, combinations/exclusions, spatial/source-container constraints and ordering. Preserve original instruction. Completed and deferred identities survive new observations; incomplete items cannot produce full success.
- A failed item may be deferred by JEV so others can finish, with a final incomplete-item report.
- Add a visible table marker region as a placement destination; detect its geometry through cameras. Support bin-to-marker tasks.
- Bin placement uses a region and observed occupied height, allows natural accumulation, validates retreat and avoids walls. Every changed scene invalidates old plans; each transfer refreshes the collision scene from measured evidence. No invisible object is treated as disappeared.
- OpenAI handles visual semantics, instance grounding, relationships and uncertain result evidence. Bundle useful questions per capture, cache same-capture results, enforce finite call budget; never supply fabricated metric poses or recovery actions.
- Refit protruding collision spheres, splitting large ones as useful; total robot model approximately 300 spheres, SDK and collision tolerances unchanged. Report coverage/overreach/self-collision/planning comparison.
- Reduce repeated work, separately report cold/warm computation, motion, perception, JEV and VLM times. Do not silently change user motion parameters.
- Keep full compact skill catalog, bounded decision state and exact original task constraints. Historical raw artifacts stay in logs.

## Implementation and validation boundaries

Existing project architecture, frozen SDK/vendor code and installed algorithm images remain intact. Local source edits are backed up; remote service is updated only after local validation and an idle-state check with source backup. Preserve confirmed scene layout. Tests cover recovery decisions differing for identical failure, specified arm, counts/exclusions, unknown visibility, stale scene rejection, marker sensing, placement occupancy and token bounds. Real JEV replay and remote RGBD/MuJoCo acceptance supplement isolated tests.
