# verify_task_completion

Check the original overall and per-object requirements against current RGB, the verified transfer/processing ledger and actual robot-posture proof. Acquire or reuse current three-camera RGB internally; this needs neither observe_scene nor new SAM3, CAD/OBB geometry or a new collision map. Empty queues and action receipts alone are not success. Dated completed requirements remain completed unless contradicted; historical images are not current visibility. Inspect required final destination, processing and return/after-delivery evidence separately. This judgment does not insert missing actions.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses selects the initial graph; Decisions selects runtime recovery. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.
汇总条件和单物体条件使用相同的输入/产物谱系及时间范围。过去的交付、加工阶段使用对应 ID 的已验证视觉释放与加工记录；已消耗输入无需仍出现在最终 RGB。当前物理状态仍由当前图像及已绑定证据判断，矛盾图像、缺失历史记录或单独 completed 状态都不能证明完成。条件原文及逐项判断保留。
