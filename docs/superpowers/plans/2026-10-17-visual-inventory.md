# Decisions visual inventory implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Existing user authorization covers implementation; reuse the requirements worktree and preserve its pending changes.

**Goal:** Resolve SAM3 proposals into a fixed visual inventory and verify transfer progress from subsequent RGBD observations using OpenAI Decisions.

**Architecture:** JEV retains task/skill control. An injected visual inventory adapter judges raw SAM3 candidates before identity grouping; initial RGB and measured positions provide references. Geometry remains RGBD-only. Release feedback and visual destination confirmation are separate evidence.

**Tech Stack:** Existing Python/httpx/Pillow/numpy; official POST /v1/decisions, gpt-6-luna. No SDK/image/dependency changes.

**Spec:** User instructions in this conversation: fixed initial full inventory plus transport subset; three cameras on updates with per-object partial visibility allowed; visual progress after release including the last object; remove superseded identity compensation after validation.

## Constraints and interfaces

- `VisualInventory.resolve(capture, detections, *, instruction, reference, entities, progress)` returns groups with fixed IDs and typed evidence; unknown never creates an identity or completion.
- A dedicated official credential configuration cannot inherit the existing third-party Responses key.
- One batched request per analyzed capture/input; cached errors do not retry. No hidden motion, new observation or JEV call inside the adapter.
- Initial candidates are grounded in SAM3 masks; raw-image coverage is checked so an incomplete mask list cannot silently become a complete inventory.
- Same-camera aliases and multiple cameras merge by visual identity. Depth remains evidence, not a 5 mm temporal matching gate.
- Final completion requires release feedback AND visual confirmation at the selected destination. Occlusion stays unknown.

## Tasks

- [x] 1. Write failing adapter tests (real request shape, initial aliases, temporal identity, partial visibility, uncertainty, cache/error handling); implement image preparation and typed Decisions client.
- [x] 2. Integrate before perception grouping and runner association; propagate initial membership and fixed IDs; test measured geometry/mask ownership and no new IDs.
- [x] 3. Update release verification, final observation and capture budget; test completed/pending/unknown and idempotency.
- [ ] 4. Verify actual initial/post-transfer images through the official API; clean replaced runtime branches, update README/skill contracts/guide; run affected/full feasible tests and review diff.

## Review focus

- Visually identical objects without identifying evidence must remain unknown.
- A box returned by a cube query must be classified from pixels, not the query label.
- A merged/incorrect mask must not become target geometry.
- Service failure must stop scene identity processing with a clear reason, not invoke the former matcher.
- No masked candidate or stale historical geometry may silently erase a current obstacle.

## Execution record

- Baseline: existing requirements worktree has prior TCP/placement changes; no changes to vendor, production or running services at plan creation.
- Official text-only endpoint probe previously succeeded; image capability and project workflow remain to be verified in this plan.
- Official image batches succeeded for initial inventory, multiple cameras, a lifted object, and post-placement identity/destination judgments. Existing captured frames were used; no fresh robotic motion was executed for this feature.
- Review reproduced and fixed stage double-counting, grasp/place identity handoff loss, silent geometry item loss, missing candidate contour input, prompt replacement, and malformed-response retrying. Tests first reproduced failures before the fixes.
- Recorded official responses replay through the revised geometry/ledger flow with unchanged green-object ID and elsewhere→at_destination. The post-placement black obstacle is not currently detected; its fixed inventory entry remains.
- Remaining gate: new contour-bearing payload and a fresh complete simulated transfer still need online acceptance. Trial returned HTTP429; final safe diagnostic reported 1453 tokens remaining, insufficient for a frame batch. No further API calls, production rollout, or physical deletion of legacy compatibility modules. See the validation report for evidence and limits.
- Final regression: Windows 1758 passed/54 skipped; isolated Ubuntu 1802 passed/10 skipped; frontend 262 passed and Vite build passed; Ruff and diff whitespace checks passed. Review fixes include low-confidence coverage rejection and idempotent collision-map retry after confirmed release. The recorded second initial count confidence was 0.65, so its saved reference is a development sample, not evidence of final initialization acceptance.
