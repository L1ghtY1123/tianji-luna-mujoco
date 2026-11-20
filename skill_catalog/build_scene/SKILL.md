# build_scene

Initialize a full map when needed or explicitly requested by observe_scene(full_collision_survey=true). A full map requires both arms at the configured pose for all three exposures and publication. With an existing complete map, patch by permanent ID even at the observation pose. Rebuild new or moved obstacles using FoundationPose for registered CAD and masked-depth TSDF for unknown shapes. Retain stationary layers with their original capture, including visible partial surfaces. Keep a moved but unseen layer stale and update only measured anonymous workspace cells.

Parameters, preconditions, effects and results are defined in contract.json. Cross-skill observations and recovery remain explicit in the graph.
