from chartrag.ingest import build_chunks, flag


def test_chunk_ids_unique_and_prefixed_with_patient(patients):
    chunks = build_chunks(patients)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c["id"].startswith(c["metadata"]["patient_id"] + "/")


def test_every_patient_has_allergy_and_medication_chunks(patients):
    ids = {c["id"] for c in build_chunks(patients)}
    for pid in patients:
        assert f"{pid}/structured/allergies" in ids
        assert f"{pid}/structured/medications-active" in ids


def test_chunks_respect_size_limit(patients):
    from chartrag import config
    for c in build_chunks(patients):
        body = c["text"].split("\n", 1)[1]
        assert len(body) <= config.MAX_CHUNK_CHARS + 200


def test_reference_range_flags():
    assert flag(8.9, "4.0-5.6") == "H"
    assert flag(3.0, "3.5-5.1") == "L"
    assert flag(45, "<30") == "H"
    assert flag(44, ">=60") == "L"
    assert flag(5.0, "4.0-5.6") == ""
