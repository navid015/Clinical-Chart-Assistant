# Risk analysis

A hazard analysis for the chart assistant, loosely following the structure of ISO 14971 (hazard, cause, harm, control, verification). It is a design exercise for a demonstration system, not a regulatory submission. Severity: 1 negligible to 5 catastrophic. Likelihood before controls: 1 rare to 5 frequent.

| ID | Hazard | Cause | Potential harm | Sev | Lik | Controls | Verification |
|---|---|---|---|---|---|---|---|
| H-01 | Another patient's data shown | Retrieval returns chunks from other charts | Wrong-patient decision; privacy breach | 5 | 3 | Patient filter in the database query; post-retrieval patient check; conversation reset on patient switch; other-patient questions refused | `test_retrieval_never_crosses_patients`, `test_unfiltered_search_would_mix_patients`, eval gate `cross_patient_leakage = 0` |
| H-02 | Fabricated finding, value or date | Model fills a gap from general knowledge | Decision based on a nonexistent result | 5 | 4 | Source-only prompt; evidence gate; citation validation; claim-level verification shown to the user | `test_invalid_citation_is_blocked`, `test_uncited_answer_is_blocked`, eval gates `faithfulness`, `keyword_recall`, `answer_correctness` |
| H-03 | Missed or misstated allergy | Allergy chunk not retrieved or summarized wrongly | Exposure to an allergen | 5 | 2 | Allergy band rendered from structured data, never from the model; dedicated allergy chunk per patient; intolerance shown differently from allergy | `test_every_patient_has_allergy_and_medication_chunks`, eval category `safety_fact` |
| H-04 | "Not documented" read as "does not have" | User or model equates missing notes with absence of disease | Missed diagnosis | 4 | 3 | Prompt rule 4; not-documented message states absence of documentation is not absence of disease | Eval category `not_documented` |
| H-05 | System gives treatment or dosing advice | User asks for a recommendation | Unsafe treatment from an unvalidated tool | 4 | 4 | Deterministic advice filter; model scope check; prompt rule 5 | `test_advice_requests_are_caught`, eval gate `refusal_accuracy` |
| H-06 | Outdated information presented as current | Older note retrieved over a newer one | Acting on a superseded plan or dose | 4 | 3 | Dates on every source and in every answer; prompt asks for the most recent value and to report conflicts with dates | Eval category `trend` |
| H-07 | Instructions hidden in a note change behavior | Prompt injection through record text | Data exposure or unsafe output | 4 | 2 | Sources marked as data, not instructions; retrieval scoped to one patient limits what could be exposed; citation validation | Eval item `oos-injection` |
| H-08 | Identifiers leak into logs | Questions contain names, MRNs, dates | Privacy breach | 3 | 4 | Redaction of roster names, MRNs, phones, emails, SSNs and dates; salted patient hash; patient prefix stripped from chunk ids | `test_audit_log_is_redacted` |
| H-09 | Over-reliance on fluent answers | Automation bias | Errors not caught by the clinician | 4 | 3 | Sources beside every answer; unverified statements listed; "not for clinical use" notice | Usability review with clinicians (planned) |
| H-10 | Silent quality regression after a change | Prompt, model or chunking change | Any of the above at a higher rate | 4 | 3 | Golden-set evaluation with quality gates on every pull request; model and pipeline versions recorded in reports and audit log | `.github/workflows/eval.yml` |

## Residual risk
With the controls above, the dominant residual risks are H-02 (subtle misstatements the verifier also misses) and H-09 (automation bias). Both require human review and a clinician usability study before any real-world use, and a larger, real-world evaluation set.
