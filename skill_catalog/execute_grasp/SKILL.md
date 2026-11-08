# execute_grasp

Execute the selected current approach/grasp trajectory, close the gripper, validate actual closure/start/collision state, then execute the planned lift. The first lift has no object attachment, as configured by the user. Preserve partial physical effects on failure; never replay them as an ordinary retry. Return measured execution/holding evidence. Attachment is prepared only after successful lift and holding confirmation.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

该动作清除 observation_pose_ready，后续要求观测位姿时由图显式安排回位。

夹爪稳定等执行超时作为 failure 返回 graph，包含原始错误和部分执行记录。超时不会放宽稳定要求，也不会清除 recovery_needed 或允许重复已下发的物理动作；无合法恢复时沿声明的结束路径停止。

## 物理回执与时间记忆

开始抬升前标记旧当前位置不再保证；夹爪反馈只记录机器人事件。抬升后机械反馈通过才建立执行层携物碰撞模型，不调用视觉夹持核验，不向物体清单写入夹持状态。物体身份、位置、加工结果由独立的观测及核验更新。后续阶段失败不抹除已发生的动作记录。

夹爪命令稳定等待使用完整窗口内的位置极差和时间加权速度均方根，保留原位置/速度阈值、窗口与超时。记录窗口统计及瞬时最大速度，接触微振不再反复重置瞬时低速计时；持续运动仍返回失败。该检查仅属于执行层，不推断物体记忆状态。


空抓恢复：execute_grasp 的失败回执明确设置 recovery_needed。Decisions 只能选择 graph 中声明的有限恢复分支。recover_failed_grasp 核对当前稳定、全闭合机械反馈和失败动作绑定；成功只清除临时执行模型及旧规划、失效目标几何，不移动、不推断物体位置。随后由 graph 显式观测、建立 observation 碰撞场景、回配置观测位姿、重新观测和规划，用独立 execute_grasp 节点重试一次。恢复或第二次抓取失败即结束；不确定反馈、限位闭合指令和已建立 attachment 均不允许此恢复。

The plan carries MotionPlanBinding for target, destination, arm, generation, observation, scene and settings. Execution validates this binding before physical commands. Grasp planning consumes a detached scene/observation snapshot throughout its bounded search.

During grasp descent and closure, the executing Link3 fingers and Link4 linkage may contact the selected target. Contact-link/environment, peer-arm, other-link/target and self-collision checks retain their configured clearance.

Gripper joint-limit violations retain actual position, command, authored limits and settling statistics in failure evidence.
