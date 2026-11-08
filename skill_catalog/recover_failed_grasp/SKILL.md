# recover_failed_grasp

Reconcile an execute_grasp failure only when current stable full-close mechanical feedback proves an empty gripper, with the same arm, target, generation and partial action receipt, and no attached collision model. No motor commands, camera capture or object-memory status inference. Clear the provisional execution model, discard old grasps/plans and invalidate affected target geometry. Graph must then observe_scene, build_scene(exclusion_state=observation), move_to_observe(configured), observe again and rebuild all grasp/placement prerequisites. Retry execute_grasp with a distinct node at most once. Any recovery or retry failure ends. Unknown feedback, limited-close commands, attached objects and other partial failures cannot use this recovery.

This skill only reconciles execution state. It does not move, retry or declare where the object is. Subsequent graph actions retain all fresh-scene and collision checks.
