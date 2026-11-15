# prepare_failure_return

Reserved terminal policy: freeze pending motion while preserving gripper commands, retain the original failure, establish a new motion epoch and classify actual observation-pose/held/empty/map state. Run exactly once after task recoveries are exhausted; never mark the task successful.

仅由系统固定失败收尾 graph 使用；不提供给 Responses 任务编排。
