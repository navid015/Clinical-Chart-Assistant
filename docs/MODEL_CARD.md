# Model card: Clinical Chart Assistant

## Summary
A retrieval-augmented question-answering system that helps a clinician review one patient's chart. It answers only from that patient's record, cites the exact note sections it used, verifies its own claims, and refuses questions outside that scope.

## Intended use
- **Users:** clinicians and care-team members reviewing a chart they already have access to.
- **Task:** find and summarize documented facts, such as results and trends, medication history, allergies, procedures, findings, and documented plans, with citations to the source notes.
- **Setting:** a demonstration built on synthetic patients. It is not validated for clinical use.

## Out of scope
- Making or suggesting diagnoses, treatments, doses or management decisions.
- Answering about any patient other than the one selected.
- Use by patients or the public, or use as the only source of truth for a clinical decision.
- Real patient data. The current system has no authentication, access control or BAA-covered hosting.

## System
| Component | Choice | Why |
|---|---|---|
| Embeddings | `text-embedding-3-small` | Low cost, adequate for a small single-patient corpus |
| Answer model | `gpt-4.1-mini`, temperature 0 | Reliable instruction following for citation format |
| Verifier and judge | same model by default, configurable | Swap in a different model to reduce self-grading bias |
| Vector store | Chroma, cosine distance | Simple, persistent, supports metadata filtering |
| Keyword search | BM25 | Drug names, lab names and numbers match better lexically |
| Fusion | Reciprocal rank fusion | Needs no score calibration between the two retrievers |

Chunking is section-aware: one chunk per note section (Assessment, Plan, Findings...) plus one chunk per structured element (allergy list, active medications, each lab test). Every chunk carries a `patient_id` used as a hard database filter.

## Safety design
1. Patient isolation enforced by the database filter, re-checked after retrieval, and tested for every question and every patient.
2. Deterministic checks before any model call: questions naming another patient and treatment-advice requests are refused.
3. A model-based scope check catches advice and off-topic requests the rules miss.
4. An evidence gate returns "not documented" without calling the model when nothing relevant is retrieved.
5. The prompt requires a citation on every factual sentence and states that absence of documentation is not absence of disease.
6. Citations are validated in code; answers citing nonexistent sources, or none, are withheld.
7. A second model call checks each claim against the sources; unverified claims are shown to the user, not hidden.
8. Allergies, problems, medications and lab trends in the side panel come straight from structured data, never from the model.
9. Every request is audit-logged with a hashed patient reference and a redacted question.

## Evaluation
51 hand-written questions across six patients (`eval/golden_set.jsonl`), each with an expected status, expected source chunks, required facts and a reference answer. Categories: direct facts, trends, reasoning ("why was X stopped"), lists, summaries, safety facts (allergies), follow-up questions, not-documented questions, and out-of-scope questions (treatment advice, other patients, off-topic, prompt injection).

Quality gates run on every pull request (`.github/workflows/eval.yml`). Current results are in the Evaluation tab and `reports/eval_report.md`.

## Limitations
- Synthetic data is cleaner and more consistent than real notes. Real charts have copy-forward text, contradictions, abbreviations and scanned documents, and would need a new evaluation.
- Six patients and 51 questions give wide confidence intervals; one failure moves a metric by several points.
- LLM-as-judge and the verifier can be wrong in both directions; spot-check them by hand.
- The rule-based advice filter is English-only and pattern-based.
- Redaction in the audit log is pattern-based and would not meet de-identification standards for real PHI. Production use would need a validated de-identification method and access-controlled logs.
- No temporal reasoning beyond what the notes state; "today" is not known to the model.

## Ethical considerations
Automation bias is the main human-factors risk: a fluent answer can be trusted more than it deserves. The interface puts the sources beside every answer, marks unverified statements, and states that missing documentation does not mean a condition is absent.
