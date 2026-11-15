# verify_place

Verify the current empty hand, then actual delivery using the bound release evidence. Container entry is success; the release square constrains robot release only. Table-region tasks retain their requested final area.

verify_place 沿用 verify_grasp 的持物与目标身份记录，默认搬运中持续持物，不新增搬运中核验。释放后先读实际关节位置：全开为空爪，中间位置仍持物；仅全闭时调用 object_held 判断。确认空爪后，place_result 在 direct_success、inferred_success、failure、unknown 中选择。桌面与入盒统一验收实际到达配置释放 XYZ 并张开夹爪；控制台小方框只约束机器人释放位置，不要求物体最终落点位于小方框内。直接确认表示看见物体放在指定桌面支撑或进入指定盒子。推测成功要求已确认持有目标、实际到达配置释放 XYZ（现有位置容差 5 mm）并释放、当前空爪，且没有交付失败的可见反证，不要求一定被遮挡。命令发出不等于实际到达；缺失绑定证据为 unknown，明确落在目的地外为 failure。当前三相机 RGB、释放前 RGB 与日期身份参考用于放置结果题，不投影释放小方框作为视觉验收边界。盒内同形物体不要求区分新旧，可依据执行证据选择推测成功。保留原始回答、执行推定身份、placement_outcome 与回执；推测入盒不伪造个体当前位姿，confidence 仅归档。

空爪判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。复用 verify_grasp 的当前接触区图与同次反馈。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持工作空间过滤、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考仍用于实例身份与放置落点核验。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。程序直接判定全开与中间位置，仅全闭时由 Decisions 看图判断；程序判定独立记录来源，不伪造模型回答。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

Release verification checks binding integrity and matching held identity target/arm/run/generation/scene. Wrong or missing identity proof cannot support inferred delivery. Empty gripper alone never proves destination success.
