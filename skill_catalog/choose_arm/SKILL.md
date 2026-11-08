# choose_arm

Choose among arms permitted by current geometry, measured robot state and offline workspace. Use the sole admissible arm directly; when several remain, Decisions proposes one. This proposal is not proof of grasp IK or trajectory feasibility. Known CAD grasps must come from the selected arm offline library. Do not impose a later processing capability arm on an earlier transfer.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

Require unique current measured target and destination geometry before numerical evaluation. Missing current identity/geometry returns unknown with observed_instance_missing_or_ambiguous. A changed viewpoint can leave a previously bound subject unassociated.

Selection performs no physical motion and is retry_safe. A graph-selected recovery may revisit this node after refreshing current evidence and restoring its prerequisites. New arm selection still invalidates placement resolution and planning; generate arm-specific grasps before motion. Physical execute_grasp/execute_place/processing actions do not inherit this repeatability.

选择问题接收同一目标此前的规划/执行失败、尝试臂和数值证据。Decisions 比较当前输入是否改变，并考虑其他合格臂。明确指定臂与数值可行域继续约束选择。
