# 清单图入口与证据回放验收

## 真实执行

实验：`962573e9ee1743db933f1b4138954115`。

任务：将左盒粉色物体放到桌面预设输入区域，不拆包，双臂回到任务起始位姿。

- 状态 succeeded，用时 195.85293680196628 秒，Responses 1 次。
- Responses 图显式产生 observe_scene、confirm_inventory、抓放、建图、回位、verify_task_completion 节点。
- 初始 observe_initial 在事件 16 返回 visual_identity_unknown；模型图已有 max_attempts=3 的 retry，第二次观测成功。
- 放置后 observe_after_placement 在事件 196 返回 visual_identity_unknown；原图重试后成功。
- 新采集对应事件 19 和 199；期间没有 Responses 修订，也没有重放抓取/释放。
- 最终 verify_task_completion 在事件 236 成功，事件 239 结束。

重试节点的 capture_mode 是模型选定的 reuse_current；代码仍执行已有的时效/版本检查，两次重试时重新采集。此实验验证的是原图中有界补观测可以实际恢复未知结果，不证明原地补拍能解除持续遮挡。该图的最终 unknown 分支选择 stop；能力接口允许更丰富的恢复，但不能声称此模型图已覆盖所有观察姿态恢复。

## 回放

对用户截图实验 `7dc80ee07b3c42efba9155ba8447fc08` 的事件 26 做浏览器检查。三路 C 输入可见，头部 R 输入可见，腕部原始 confidence 0.50/0.66/0.71 可见，原始 RGB 切换和实际调用详情可打开，无页面脚本错误。读取归档不修改机器人或实验。

只展示所选结果版本之前最近一次身份相关调用，不把之后的调用混到旧结果中。重复的模型输入文件在三相机卡片共享请求缓存。所有图片来自真实请求归档；旧结果 SVG 未重写。

本地证据：`data/observation-policy-live-events.json`、`data/observation-policy-browser/result.json`、`data/observation-policy-browser/three-camera-provenance.png`。

## 验证边界

完整后端回归：3426 passed / 58 skipped；前端：371 passed；Vite build 成功，仅保留既有大 chunk 提示。后续增加清单版本保护及否定/未知结果区分后，控制器、图执行、模型集成及最终位姿相关回归 74 passed，含“核验后重新观察清单，旧完成证明失效”和初始 unknown/no 分支用例。无提交、无 PR。

冻结身份请求的逐 ID 谓词探针改善了粉色物体腕部判断，但没有稳定解决两个蓝盒歧义，因此未将该探针变成上线逻辑，也未修改身份门槛。当前改动修复了输入/输出可见性和图外观测无法恢复的问题，不能宣称全部跨相机 ID 关联已解决。

回归中原有 mock 图曾依赖执行器偷偷进行初始观测/确认。测试模型夹具已改为显式输出这些节点；历史中途图测试仍使用其当时已有的确认状态。未通过修改生产逻辑恢复旧的图外流程。
