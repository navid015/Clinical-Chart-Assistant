import json

from chartrag import config
from chartrag.pipeline import MESSAGES


def test_answered_with_valid_citations(assistant):
    r = assistant.ask("P001", "How has her hemoglobin A1c changed?")
    assert r.status == "answered"
    assert r.citation_map and all(cid.startswith("P001/") for cid in r.citation_map.values())
    assert r.verification and r.verification["score"] == 1.0


def test_other_patient_is_refused_before_retrieval(assistant):
    r = assistant.ask("P001", "What was Raymond Okafor's ejection fraction?")
    assert r.status == "out_of_scope" and r.reason == "other_patient"
    assert r.retrieved == []


def test_advice_is_refused(assistant):
    r = assistant.ask("P002", "Should we increase his furosemide?")
    assert r.status == "out_of_scope" and r.reason == "clinical_advice"


def test_irrelevant_question_hits_evidence_gate(assistant):
    r = assistant.ask("P004", "zqxv wplk")
    assert r.status == "not_documented"
    assert r.answer == MESSAGES["not_documented"]


def test_empty_question(assistant):
    assert assistant.ask("P001", "   ").status == "out_of_scope"


def test_unknown_patient_returns_error_not_crash(assistant):
    assert assistant.ask("P999", "anything").status == "error"


def test_invalid_citation_is_blocked(assistant, monkeypatch):
    from chartrag import llm
    real = llm.chat
    monkeypatch.setattr(llm, "chat", lambda m, **k: "A1c was 6.9% [S9]." if "TASK: ANSWER" in m[0]["content"] else real(m, **k))
    r = assistant.ask("P001", "What is her latest A1c?")
    assert r.status == "blocked" and r.reason == "invalid_citations"


def test_uncited_answer_is_blocked(assistant, monkeypatch):
    from chartrag import llm
    real = llm.chat
    monkeypatch.setattr(llm, "chat", lambda m, **k: "Her diabetes is well controlled overall." if "TASK: ANSWER" in m[0]["content"] else real(m, **k))
    assert assistant.ask("P001", "Is her diabetes controlled?").status == "blocked"


def test_audit_log_is_redacted(assistant):
    assistant.ask("P002", "Does Raymond Okafor (SYN-204417) have allergies? Seen 2025-11-02.")
    line = (config.LOG_DIR / "audit.jsonl").read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert "Okafor" not in line and "SYN-204417" not in line and "2025-11-02" not in rec["question_redacted"]
    assert "P002" not in json.dumps(rec["retrieved"]) and rec["patient_ref"] != "P002"


def test_paraphrased_not_documented_is_not_blocked(assistant, monkeypatch):
    from chartrag import llm
    real = llm.chat
    monkeypatch.setattr(llm, "chat", lambda m, **k: "The provided sources do not mention a colonoscopy." if "TASK: ANSWER" in m[0]["content"] else real(m, **k))
    assert assistant.ask("P001", "Has she had a colonoscopy?").status == "not_documented"


def test_marker_after_preamble_is_detected(assistant, monkeypatch):
    from chartrag import llm
    real = llm.chat
    monkeypatch.setattr(llm, "chat", lambda m, **k: "Checked all notes.\nNOT_DOCUMENTED: colonoscopy" if "TASK: ANSWER" in m[0]["content"] else real(m, **k))
    r = assistant.ask("P001", "Has she had a colonoscopy?")
    assert r.status == "not_documented" and "colonoscopy" in r.answer
