# ground_task

Ground the user's roles on a frozen three-camera capture. Associate each wrist to the head with calibrated geometry and commit confirmed scene instances. A fresh observation with history requires a Decisions identity verdict. Decisions binds each new task role to current registered IDs using the original instruction, Responses hints and measured robot-frame positions. Unique instances remain bound across moves. Unknown roles preserve committed physical IDs and request graph recovery.

Inputs, preconditions and outcomes remain authoritative in contract.json. Unknown requires an explicit graph branch.
