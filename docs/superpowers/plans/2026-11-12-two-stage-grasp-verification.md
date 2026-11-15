# 两阶段抓取验证实施计划

采用 executing-plans 在当前项目执行，遵循 AGENTS.md 的原目录与单代理要求。

## 用户确定的行为

Lift 完成后先判断夹爪是否抓住物体。确认抓住后，再判断所持物体是否为 target。两个问题使用二值谓词并复用同一采集。目标身份确认后建立 attachment。

## 实现与验收

- [x] 在 robot_skills/grasp_verification.py 定义持有谓词和目标身份谓词。
- [x] 在 runner.py 顺序调用，第一项否定时跳过身份问题，回答无效或平分时保留未完成验证。
- [x] 在 grasp_recovery.py 使用第一项否定证据确认空抓，保留目标、臂、代次、动作版本及 attachment 检查。
- [x] 同步 verify_grasp 的合同、技能说明、编排提示与现行文档。
- [x] 验证空抓、正确目标、错误目标、两阶段分别未知、缺失参考、单次采集和 attachment 时序。
- [ ] 校验固定子图和 Responses 单次编译边界，部署后完整仿真搬运并检查两条实际请求的采集一致性。

## 范围

本次修改落实用户确定的抓取验证顺序。跨相机选项、新实例发现和 CAD 流程先依据当前实现回答，后续策略由用户继续确定。

## 实测记录

单次 Responses 编译边界、分支检查、协议检查、16 项关联与恢复检查、部署哈希和浏览器回放检查通过。
完整搬运验收仍待通过。三轮记录位于 data/runs/two-stage-grasp-20261112/acceptance-1.json 至 acceptance-3.json。首轮新增 scope 未接入白名单，已修复并补测实际输入入口。后两轮出现夹持图被判为空抓。
简化问句、添加前次图像、仅使用腕部图的离线对照均未纠正两个夹持样本，分别保存在 prompt-comparison.json、history-comparison.json 和 wrist-comparison.json。
第二问成功及目标确认后的完整搬运路径已做隔离分支检查，尚未在本次端到端仿真中通过。
