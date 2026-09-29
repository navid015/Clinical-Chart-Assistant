from chartrag.ingest import build_chunks

VALID = {"answered", "not_documented", "out_of_scope"}


def test_golden_set_is_well_formed(golden, patients):
    ids = [g["id"] for g in golden]
    assert len(ids) == len(set(ids))
    for g in golden:
        assert g["patient_id"] in patients, g["id"]
        assert g["expected_status"] in VALID, g["id"]
        if g["expected_status"] == "answered":
            assert g.get("expected_sources") and g.get("reference_answer"), g["id"]


def test_expected_sources_exist_and_belong_to_patient(golden, patients):
    """Catches drift: if a note is renamed or re-sectioned, the golden set must be updated too."""
    chunk_ids = {c["id"] for c in build_chunks(patients)}
    for g in golden:
        for s in g.get("expected_sources", []):
            assert s in chunk_ids, f"{g['id']}: unknown source {s}"
            assert s.startswith(g["patient_id"] + "/"), g["id"]


def test_golden_set_covers_safety_categories(golden):
    cats = {g["category"] for g in golden}
    assert {"clinical_advice", "other_patient", "off_topic", "not_documented", "safety_fact"} <= cats
