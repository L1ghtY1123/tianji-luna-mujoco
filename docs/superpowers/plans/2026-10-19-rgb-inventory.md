# RGB inventory and mask identity implementation

Approved by the user on 2026-10-19. Implement inline with executing-plans.

1. Responses Structured Outputs creates the initial tabletop entity inventory from head RGB and the instruction. Application code allocates permanent IDs. Keep table, containers, obstacles, transport members and regions; SAM3 detections do not define membership or inventory size.
2. SAM3 uses inventory descriptions to produce candidate masks. Decisions sees RGB originals, reference boxes/crops, current numbered boxes/crops and task state; it maps candidates to permanent IDs and checks transfer status. No depth arrays, world geometry or mask contour input. The application binds masks to IDs. Identical objects are separate entries, and uncertainty remains explicit.
3. Preserve all semantic entities and original references when current masks/depth are missing. Collision map publication still requires current geometry for every collision entity. Keep table reuse, JEV outer policy, metric placement and release checks.
4. Reuse visual call archival for both endpoints; update skills and guide. Remove superseded candidate-based initial discovery, heuristic class exploration and identity allocation.
5. Add failing contract/integration tests, implement, run focused and full suites. Verify official account access and saved real RGB separately. Deploy only after actual image inference and isolated cycle acceptance; record any external block precisely.

Baseline: `.superpowers/sdd/2026-10-19-rgb-inventory/baseline`. Existing worktrees are dirty; preserve unrelated changes and do not commit or copy whole trees.

## Execution ledger

- Investigation: candidate code currently discovers from SAM3 masks. Production still uses earlier implementation; candidate identity changes were never deployed. The new implementation must not be described as live until deployment is verified.
- Ruling: use the existing dedicated official OpenAI configuration, never the unrelated OPENAI_API_KEY/provider configuration. No SDK dependency changes; existing HTTPX transport with documented public API.
- Ruling: slot XY is a TCP anchor, not an object-centre containment requirement. Visual destination evidence cannot certify millimetre TCP placement.

- Implemented RGB-first discovery, fixed-ID binding, missing-geometry ledger retention, initial semantic image labels, shared endpoint audit and JEV/doc contracts. Removed candidate-based ID allocation, initial coverage counting and class/synonym expansion.
- RED: test_rgb_inventory.py initially 4 failed/1 passed; GREEN after adapters/integration. Initial missing-SAM label regression failed first, then passed. Visual HTTP conclusion tests failed first, then passed.
- Full Python suite: 1670 passed,54 skipped (91.30s); frontend:271 passed. Two later conclusion tests passed focused; final suite rerun pending review fixes.
- Actual network: Responses 200,10.239s,1 image,6 entities,1572 total tokens. Fresh SAM3:4 inventory prompts each returned1 candidate. Decisions:429,type=tokens,code=rate_limit_exceeded,1.579s,23 images,16 questions; no identity/progress answers. No retries, no full robot cycle, no production switch.
- Ruling: keep the candidate isolated because new Decisions inference and cycle acceptance cannot complete under the observed token limit. This leaves production's old vision behavior active; it does not justify claiming the new flow is live.

- Final review: P1 initial inventory loss after downstream failure and P2 RGB-box top-edge ordering reproduced, fixed with RED→GREEN tests. Initial failures now retain IDs/reference in Runner, show semantic labels, and a later observation uses three views without another Responses discovery. Numbering shares existing row grouping. No further important findings.
- Full suite after review fixes:1676 passed,54 skipped,89.27s. Added a subsequent RED→GREEN assertion for semantic labels on the same failure path;39 focused tests passed; final full delivery run:1676 passed,54 skipped,89.55s. Frontend271 passed; Ruff and diff checks passed.
- Synchronized70 candidate files to Ubuntu after checking core-source baseline hashes; backed up existing files and verified all70 SHA256 values. Production remains unchanged. An additional isolated Linux pytest invocation could not start because the existing runtime image has no pytest; no dependencies were installed.
- Actual Decisions recheck using the runtime's host network:429 in1.605s; limit100000,used95161,remaining4839,requested8366 tokens,Retry-After91420s. An earlier bridge-network diagnostic returned a transport error and no HTTP response. No identity/progress answers and no transfer-cycle acceptance.
- Ruling: retain this task's baseline, logs and image evidence because changes are uncommitted and the worktree contained prior edits; deleting scratch or committing the mixed tree would lose provenance or include unrelated work. No merge/push/cleanup requested. Candidate implementation is complete; online acceptance and production switch remain pending the external token limit.
