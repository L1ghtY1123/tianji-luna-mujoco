# Decisions outcomes implementation plan

**Goal:** Preserve why a judgment was unknown and route real contradictions as failures through the existing Responses graph.

**Architecture:** Keep the version-2 graph, skill boundaries and probability thresholds. Use the existing answer parsers, diagnostic records and router callback; remove the callback's lossy bool/None conversion. No dynamic graph edits, new planner or SDK changes.

**Spec:** The user's request to continue fixing failure recovery and Decisions unknown outcomes; AGENTS.md remains the project contract. Work inline, without subagents, worktrees or test directories.

**Validation:** Direct executable regression probes, Python syntax/lint, frontend build if changed, synchronized hashes and an actual unpack simulation. Probes must cover refusal without probability, invalid responses, low/high probability, mixed contradiction/unknown, immutable recovery routes and raw evidence retention. A successful run is not a success-rate estimate.

- [x] Preserve typed refusals in shared predicate parsing and update affected consumers without inventing probabilities.
- [x] Distinguish uncertain probability, low choice confidence, explicit unknown and refusal in diagnostics; preserve operation proposals.
- [x] Carry the structured router result and evidence to subsequent declared branches and replay; reject invalid callback results.
- [x] Give a definite contradiction precedence over another missing criterion in conjunction-based acceptance.
- [x] Align current prompt, contracts and documentation, keeping recovery graph-authored and bounded.
- [x] Run focused regressions, synchronize, and verify the complete task with timing and failure evidence.

The actual first run exposed a compound recovery condition and a mixed historical/current completion condition. Refined both field descriptions and the prompt, removed the redundant aggregate completion predicate, and checked frozen evidence before the final full run. See ../../decision-outcomes-validation.md for results and remaining limits.
