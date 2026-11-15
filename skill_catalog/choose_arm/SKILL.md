# choose_arm

Qualify one arm whose position workspace contains both the measured source anchor and the estimated release TCP, using visible object height, measured source support and destination table or container rim plus the configured release margin. Recheck the exact release goal with actual grasp TCP feedback during plan_place. No common arm means failure. Pose IK and carried path remain numerical planning requirements. Lock the chosen arm for this transfer; recovery cannot change it. No Decisions or handover.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。
