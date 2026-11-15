# return_after_failure

Reserved terminal numerical dual-arm return from actual joints to configured observation posture. Preserve verified holding and its attachment. For confirmed empty grippers, validate opening after joint return, then open and verify actual finger feedback. Plan and validate full robot, carried object, world/self collision and joint limits using cuRoboV2-Fast. Stop after measured return and preserve the original task failure. Unsafe or unavailable return is explicitly blocked.

仅由系统固定失败收尾 graph 使用。
