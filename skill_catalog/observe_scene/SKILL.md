# observe_scene

Acquire or reuse one verified shared head-and-wrist RGBD frame set. Request full_collision_survey only for a full collision survey at the configured dual-arm observation pose. Compare each wrist with the head, then ask Decisions to identify fresh physical groups against dated references and action history. An identical frozen capture can reuse its verified masks and IDs. Retain unseen identities in memory, and route unresolved roles through the graph. Publish accepted geometry, using a current pose for known CAD.

Inputs, preconditions and outcomes remain authoritative in contract.json. Unknown requires an explicit graph branch.
