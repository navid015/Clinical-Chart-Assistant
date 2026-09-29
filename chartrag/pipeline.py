"""
The request pipeline. Each step can end the request early with a clear status:

  question
    -> [1] other-patient check (deterministic)          -> out_of_scope
    -> [2] advice-request rules (deterministic)         -> out_of_scope
    -> [3] scope triage + follow-up rewrite (LLM, JSON) -> out_of_scope
    -> [4] patient-scoped hybrid retrieval
    -> [5] evidence gate (no LLM call if nothing relevant) -> not_documented
    -> [6] grounded answer generation
    -> [7] citation validation (deterministic)          -> blocked
    -> [8] claim-level verification (LLM)               -> flagged in the UI
    -> [9] audit log
"""
import re
import time
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel

from . import audit, config, llm
from .guardrails import check_citations, is_advice_request, mentions_other_patient
from .prompts import ANSWER_SYSTEM, SCOPE_SYSTEM
from .records import Patient, get_patients
from .retrieval import RetrievedChunk, Retriever
from .verify import format_sources, verify_answer

MESSAGES = {
    "other_patient": ("This question refers to a different patient. To keep records separate, I only answer "
                      "about the patient currently selected. Switch patients to ask about someone else."),
    "clinical_advice": ("I can't make treatment, dosing or diagnostic recommendations. I can tell you what the "
                        "record documents, such as current medications, recent results, or the plan a "
                        "clinician wrote."),
    "off_topic": "I only answer questions about the selected patient's medical record.",
    "not_documented": ("I couldn't find this in the available record for this patient. Absence of "
                       "documentation does not confirm that something did not happen."),
    "blocked": ("I withheld the generated answer because its citations did not match the retrieved sources. "
                "Review the sources below directly, or rephrase the question."),
    "error": "Something went wrong while answering. The request was logged; please try again.",
}


NOT_FOUND_RE = re.compile(r"\b(not (documented|mentioned|found|recorded|available)|no (documentation|record|mention)|"
                          r"(do|does) not (contain|mention|include|document))\b", re.I)


class ScopeDecision(BaseModel):
    category: str
    standalone_question: str
    reason: str = ""


@dataclass
class AnswerResult:
    status: str                      # answered | not_documented | out_of_scope | blocked | error
    answer: str
    reason: str | None = None
    standalone_question: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    cited: list[RetrievedChunk] = field(default_factory=list)
    citation_map: dict[int, str] = field(default_factory=dict)   # S-number -> chunk id
    citation_coverage: float | None = None
    verification: dict | None = None
    blocked_cross_patient: int = 0
    latency_s: float = 0.0
    request_id: str = ""


def _history_text(history: list[dict] | None, turns: int = 6) -> str:
    if not history:
        return "(none)"
    lines = []
    for m in history[-turns:]:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            lines.append(f"{m['role']}: {content.strip()}")
    return "\n".join(lines) or "(none)"


class ChartAssistant:
    def __init__(self, retriever: Retriever | None = None, patients: dict[str, Patient] | None = None):
        self.patients = patients or get_patients()
        self.retriever = retriever or Retriever()
        self.roster = [{"id": p.id, "name": p.name, "mrn": p.mrn} for p in self.patients.values()]

    def ask(self, patient_id: str, question: str, history: list[dict] | None = None) -> AnswerResult:
        start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        try:
            result = self._ask(patient_id, str(question or "").strip(), history)
        except Exception as exc:
            result = AnswerResult("error", MESSAGES["error"], reason=f"{type(exc).__name__}: {exc}"[:300])
        result.request_id = request_id
        result.latency_s = round(time.perf_counter() - start, 3)
        audit.log_request(result, patient_id, question, self.roster)
        return result

    def _ask(self, patient_id: str, question: str, history) -> AnswerResult:
        if patient_id not in self.patients:
            raise KeyError(f"Unknown patient {patient_id}")
        patient = self.patients[patient_id]
        if not question:
            return AnswerResult("out_of_scope", "Please type a question about the selected patient.", "empty")

        # [1] and [2]: deterministic checks, no model call needed
        if mentions_other_patient(question, patient_id, self.roster):
            return AnswerResult("out_of_scope", MESSAGES["other_patient"], "other_patient")
        if is_advice_request(question):
            return AnswerResult("out_of_scope", MESSAGES["clinical_advice"], "clinical_advice")

        # [3] scope triage + standalone rewrite for follow-up questions
        scope = llm.chat_json([
            {"role": "system", "content": SCOPE_SYSTEM},
            {"role": "user", "content": f"Conversation so far:\n{_history_text(history)}\n\nCurrent question: {question}"},
        ], ScopeDecision)
        standalone = scope.standalone_question.strip() or question
        if scope.category != "chart_question":
            reason = scope.category if scope.category in MESSAGES else "off_topic"
            return AnswerResult("out_of_scope", MESSAGES[reason], reason, standalone)
        if mentions_other_patient(standalone, patient_id, self.roster):
            return AnswerResult("out_of_scope", MESSAGES["other_patient"], "other_patient", standalone)

        # [4] retrieval, hard-scoped to this patient
        search = self.retriever.search(patient_id, standalone)
        chunks = search.chunks
        base = dict(standalone_question=standalone, retrieved=chunks,
                    blocked_cross_patient=search.blocked_cross_patient)

        # [5] evidence gate
        if not chunks or (search.max_similarity < config.MIN_SIMILARITY and search.max_bm25 <= 0):
            return AnswerResult("not_documented", MESSAGES["not_documented"], "no_relevant_sources", **base)

        # [6] grounded generation
        header = f"Patient: {patient.id}, {patient.age()}-year-old {patient.sex.lower()}."
        raw = llm.chat([
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{header}\n\nSources:\n{format_sources(chunks)}\n\nQuestion: {standalone}"},
        ]).strip()

        if "NOT_DOCUMENTED" in raw.upper():
            detail = raw.upper().split("NOT_DOCUMENTED", 1)[1]
            detail = raw[len(raw) - len(detail):].lstrip(":").strip()
            text = MESSAGES["not_documented"] + (f"\n\nSearched for: {detail}" if detail else "")
            return AnswerResult("not_documented", text, "model_not_documented", **base)

        # [7] citation validation
        check = check_citations(raw, len(chunks))
        if not check.cited and NOT_FOUND_RE.search(raw):
            # The model said "not documented" in its own words: treat it conservatively as not documented
            return AnswerResult("not_documented", MESSAGES["not_documented"], "model_not_documented", **base)
        if not check.ok:
            reason = "invalid_citations" if check.invalid else "no_citations"
            return AnswerResult("blocked", MESSAGES["blocked"], reason, citation_coverage=check.coverage, **base)

        cited_nums = sorted(check.cited)
        result = AnswerResult("answered", raw, None, citation_coverage=check.coverage,
                              cited=[chunks[n - 1] for n in cited_nums],
                              citation_map={n: chunks[n - 1].id for n in cited_nums}, **base)

        # [8] claim-level verification against the sources the model actually saw
        if config.VERIFY_ANSWERS:
            result.verification = verify_answer(raw, chunks).to_dict()
        return result
