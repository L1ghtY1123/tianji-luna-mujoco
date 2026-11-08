# Online policy boundaries implementation plan

**Goal:** Responses owns the complete task graph; Decisions evaluates evidence and selects only declared edges. Final dual-arm observation posture remains required.

**Architecture:** Keep the existing version-2 graph executor and fixed robot skills. Remove dynamic graph synthesis and per-delivery observation policy. Validate task requirements, skill prerequisites, result routes and completion receipts before reducing prompt duplication.

**Tech stack:** Python/Pydantic, Responses, typed Decisions, existing MuJoCo services and web replay.

**Spec:** The user's six approved boundary rules, reflected in AGENTS.md. Execute inline; the repository forbids subagents and additional worktrees.

## Constraints and acceptance

- Preserve skill collision, freshness, identity, holding and VLA input checks. Do not modify SDKs or algorithm services.
- No tests directories, replacement framework, hidden recovery actions or fabricated yes verdicts.
- Remove retired code, metadata, replay entrypoints and documentation together; propagate deletions to Ubuntu.
- A negative semantic judgment is failure; insufficient evidence is unknown; partial physical execution cannot be replayed.
- Validate declared failure routing and graph immutability using direct executable probes, then syntax/lint/build and actual simulation.
- Compare full versus compact skill descriptions with the same task and initial evidence. Report compilation validity, payload size and limitations; one run does not establish a success rate.

## Implementation

- [x] Remove RecoveryPlanner and graph patching; expose the failed result to authored routers and enforce declared routes.
- [x] Remove universal intermediate return/observe receipts, retain final feedback verification and VLA prerequisites.
- [x] Align skill outcomes, structured descriptions, graph rules and preflight validation; remove redundant prompt fields.
- [x] Check current replay/docs and eliminate obsolete dynamic-recovery claims.
- [x] Verify locally, synchronize exact changes/deletions, compare prompt variants and run representative simulation.
