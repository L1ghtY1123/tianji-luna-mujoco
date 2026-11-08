# Replay and question audit

**Goal:** Explain every runtime unknown source, audit Responses/Decisions wording and make actual graph execution and collision stages readable.

**Architecture:** Keep one initial Responses graph and fixed skills. Do not create recovery nodes at runtime, lower thresholds, change physics or hide executed failures. Reuse archived evidence and existing replay components. No subagents or test directories.

- [x] Audit recent failed/successful runs and all question constructors/shared prompt transforms. Record exact source, type, evidence and output rule.
- [x] Fix demonstrated ambiguous wording and misclassified results; expose unknown provenance without guessing model reasoning.
- [x] Show the mainline plus executed paths by default, with full authored graph available. Use a regular vertical layout and remove retired-framework text.
- [x] Archive mapping stage/scope/exclusion metadata; show distinct persistent object colors, solid collision surfaces and excluded wireframes with a legend.
- [x] Verify parser/routing/layout invariants, inspect actual rendered replay, build, synchronize and run the complete simulation.

Acceptance includes numerical and evidence failures remaining failures, no fabricated model answers, archived scene versions preserved, global/local updates distinguished from retained world geometry, and genuine failure paths visible in replay.
