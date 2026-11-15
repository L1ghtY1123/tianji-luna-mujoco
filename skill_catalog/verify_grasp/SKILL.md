# verify_grasp

verify_grasp 在 Lift 后先读取同次实际关节位置：全开为未持物，中间位置为持物；仅全闭时调用一次 Decisions，输入当前夹持接触区原始分辨率 RGB 和实测反馈，不调用 SAM3。明确未持物允许空抓恢复；请求异常、拒答或证据不足保留 unknown。确认持物后，程序核对连续抓取记录的目标、执行臂、运行、代次、来源观测、场景及动作版本，继承抓取前已确认目标的永久 ID，再建立 attachment。identity_binding 明确标记 continuous_grasp_execution 和 visual_identity_verified=false；这属于执行过程推定身份。执行记录不一致时保留持物结果与身份未确定状态，不建立 attachment。历史参考保留用于抓取前关联与放置核验，正常主线不再单独做抓后身份题。

实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。当前局部图由同次 TCP 和标定投影，投影不可用时使用完整腕图。持物题不发送历史整图。夹爪反馈不代表夹持力，程序按实测关节位置先判定全开与中间位置，仅全闭时调用 Decisions。

Continuous identity inheritance additionally requires a new post-Lift capture, same-generation dated Decisions reference with request provenance, and unchanged run/arm/target/scene/action bindings. Identity remains inferred from execution, not visually re-identified.
