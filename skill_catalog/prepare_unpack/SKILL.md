# prepare_unpack

If the supplied completion_requirements.after_each_object requires return and inventory observation after delivery, the verified input release must have its matching post-delivery observation receipt before this preparation starts. This includes intermediate input-area delivery. A future product return cannot satisfy it. The graph chooses the explicit return/observation nodes; this skill never inserts them.

Plan and execute both arms to the configured vla_unpack starting joints using the current measured collision scene. Both arm goals belong to that capability configuration. This does not process the object, verify a product, choose an arm for other transfers or force an observation return afterward.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
