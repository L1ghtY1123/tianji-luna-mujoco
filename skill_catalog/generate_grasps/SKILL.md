# generate_grasps

Generate candidates for the bound target and selected arm. Known CAD loads that arm offline grasps in the source CAD frame and transforms them through the current FoundationPose pose; it does not call GraspGen. Unknown OBB objects use GraspGen online. Keep current observation/model/frame provenance. Generation does not prove IK, collision clearance, contact or path feasibility. A CAD/offline-library failure is explicit, not an automatic switch to GraspGen.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
