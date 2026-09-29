import pytest

from chartrag.guardrails import check_citations, is_advice_request, mentions_other_patient

ROSTER = [{"id": "P001", "name": "Evelyn Hartwell", "mrn": "SYN-100381"},
          {"id": "P002", "name": "Raymond Okafor", "mrn": "SYN-204417"}]


@pytest.mark.parametrize("q", [
    "Should we increase her empagliflozin to 25 mg?",
    "What dose of prednisone should I prescribe?",
    "Recommend a pain medication for her.",
    "What should we give her for the pain?",
    "Should I stop his apixaban before surgery?",
])
def test_advice_requests_are_caught(q):
    assert is_advice_request(q)


@pytest.mark.parametrize("q", [
    "Why was metformin reduced?",
    "What dose of metformin is she on?",
    "What was the plan at the last visit?",
    "When was the dose of furosemide increased?",
])
def test_chart_questions_are_not_flagged(q):
    assert not is_advice_request(q)


def test_other_patient_by_name_mrn_or_id():
    assert mentions_other_patient("What is Raymond's EF?", "P001", ROSTER)
    assert mentions_other_patient("compare with okafor", "P001", ROSTER)
    assert mentions_other_patient("look up SYN-204417", "P001", ROSTER)
    assert mentions_other_patient("same as P002?", "P001", ROSTER)


def test_own_name_is_allowed():
    assert not mentions_other_patient("Does Evelyn have allergies?", "P001", ROSTER)
    assert not mentions_other_patient("What is her A1c?", "P001", ROSTER)


def test_citation_check_valid():
    c = check_citations("A1c was 6.9% on 2026-03-11 [S1]. eGFR was 46 on the same date [S2].", 3)
    assert c.ok and c.cited == {1, 2} and c.coverage == 1.0


def test_citation_check_rejects_out_of_range_source():
    c = check_citations("A1c was 6.9% on 2026-03-11 [S7].", 3)
    assert not c.ok and c.invalid == {7}


def test_citation_check_rejects_uncited_answer():
    c = check_citations("Her A1c has improved considerably over the last two years.", 3)
    assert not c.ok and c.coverage == 0.0
