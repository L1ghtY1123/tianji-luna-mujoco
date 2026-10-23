# Frozen attachment geometry

Experiment: `ef356390f2a04c2b9d025532a4369ec5`, `Algorithms.plan_place` event 75.

`luna_first_carried_surface.npz` contains only geometric regression data:

- 2,119 original registered RGBD points followed by 888 explicitly labeled height-completion points, expressed in the actual grasp TCP frame.
- Forty-two own-gripper collision spheres transformed from the project URDF/YAML using recorded driver feedback. Existing Link3 grasp-contact pairs are excluded; Link4 remains included.
- The recorded 5 mm measurement uncertainty and 1 mm attachment margin.

The old fixed-TCP-axis partition produced a 0.418 mm Link4 overlap. Local principal-direction partitions at the same 128-sphere budget retain all padded points and leave at least 1.201 mm separation. This fixture tests geometric fitting, not world collision checking or motion success.
