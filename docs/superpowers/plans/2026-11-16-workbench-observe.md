# Single workbench and observation startup

Execute inline under the repository constraints: no agents, worktrees, new images, SDK edits or real robot motion. The user's five explicit requests authorize the implementation.

Design: retain one Luna HTML entry and switch between mounted task, scene editor, collision, workspace and replay panels. Reuse existing feature modules with explicit mount/visibility lifecycles, preserve running tasks and stop inactive previews. Base's renderer remains the robot viewer; collision diagnostics supply the same explicit joints/open-gripper configuration used for the spheres through Base's existing stateless display API. These are display inputs, never hardware commands or task evidence.

Simulation initializes directly at the observe configuration before feedback/cameras/controllers consume its state, then holds that pose using the existing controller. Base publishes the same built-in observe preset. Startup of a real Base console continues to use measured hardware state and does not itself issue movement. Verify the existing user's observation preset before choosing joint values.

- [x] Reproduce missing robot geometry and trace observation preset storage; verify real/simulation launcher commands and current runtime configuration.
- [x] Add the shared observe pose as Base preset and portable Simulation startup configuration. Verify first feedback, startup hold, limits and scene recompilation without robot reset.
- [x] Replace separate Luna HTML entry points with in-page navigation; integrate existing feature modules with scoped DOM, history/deep links and inactive-panel cleanup. Preserve task socket/state, scene drafts and replay selection.
- [x] Display Base robot geometry and cuRobo spheres from matching explicit joint/gripper state; verify model visibility, opacity/contact controls and no viewer/hardware state mutation.
- [x] Build, run focused temporary checks and established Simulation regressions, sync removals, deploy only owned simulation services, exercise browser navigation and observe startup, publish exact start/stop commands and measured limits.

Review focus: duplicate DOM IDs and leaked global styles; asynchronous loads completing after a panel is hidden; paused/hidden viewers continuing to poll; cached diagnostic geometry inconsistent with spheres; startup feedback briefly showing zero pose; preset/configuration values diverging across repositories. Tests remain outside Luna/Base, with Simulation tests in its existing test directory.
