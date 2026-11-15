# move_to_observe

Explicitly plan and execute a configured observation pose or declared wrist alternative. For recovery the program must find missing current identity/geometry/image evidence, confirmed empty hands and a genuinely different pose. Failure/low probability alone is insufficient. All world/self collision, map freshness and execution feedback checks remain required. The graph owns subsequent observation.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。
