# capture_scene

Acquire or reuse one synchronized stable three-camera RGBD group through Base. This skill only captures and archives sensor evidence. It does not segment, assign IDs, build a map or move the robot. For missing initial grounding evidence, explicitly use capture_mode=fresh before ground_task.

Inputs, preconditions and outcomes are authoritative in contract.json. Unknown requires an explicit graph branch; no hidden capture, motion or retry.
