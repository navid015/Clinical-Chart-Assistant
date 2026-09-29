# Five-minute demo

1. **Structured data is deterministic.** Select Dolores Kincaid (P004). Point out that the allergy band, problem list with ICD-10 codes, medications and the lab chart come directly from the structured record, not the model. Codeine is marked as an intolerance, latex as an allergy.
2. **Grounded answer with evidence.** Ask "Were there any complications after surgery?" Show the citations, the matching source cards, and the verification line.
3. **Temporal reasoning.** Switch to Evelyn Hartwell (P001). Ask "How has her A1c changed over time?", then the follow-up "And her kidney function?" to show the follow-up rewrite.
4. **Not documented.** Ask "Has she had a colonoscopy?" Explain why the answer says the record doesn't document it, rather than "no".
5. **Refusals.** Ask "Should we increase her empagliflozin to 25 mg?" (advice, caught by rules without a model call) and "What is Raymond Okafor's ejection fraction?" (other patient, caught before retrieval).
6. **Engineering.** Open the Evaluation tab: gates, per-category results, items to review. Then show the GitHub Actions runs: offline unit tests on every push, the eval gate on pull requests, and auto-deploy to this Space.
7. **Governance.** Open the Risk analysis tab and trace one hazard, H-01 wrong patient, to its controls and to the tests that verify them.
