# build_scene

Rebuild collision geometry across the full workspace from the current registered observation on every call. Use measured table cuboids, currently registered CAD, and current TSDF/ESDF for all remaining visible geometry. Anonymous obstacles require no identity association. Apply the strict workspace mask. Replace previous object geometry and TSDF cells; retain permanent identity and operation history separately in the database. There is no local/global selector or object subset. Unknown target surfaces retain current capture provenance for GraspGen. A full rebuild does not certify unseen space as empty.

Parameters, preconditions, effects and results are defined in contract.json. Cross-skill observations and recovery remain explicit in the graph.
