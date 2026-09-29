from chartrag.retrieval import expand_query, tokenize


def test_retrieval_never_crosses_patients(retriever, patients, golden):
    """Every golden question, asked for EVERY patient, returns only that patient's chunks."""
    questions = {g["question"] for g in golden}
    for pid in patients:
        for q in questions:
            res = retriever.search(pid, q)
            assert all(c.patient_id == pid for c in res.chunks), (pid, q)
            assert res.blocked_cross_patient == 0


def test_unfiltered_search_would_mix_patients(retriever):
    """Control: without the patient filter, results do mix patients, so the filter is doing real work."""
    from chartrag import llm
    res = retriever.collection.query(query_embeddings=llm.embed(["medications allergies"]), n_results=30)
    assert len({m["patient_id"] for m in res["metadatas"][0]}) > 1


def test_abbreviation_expansion():
    assert "hemoglobin a1c" in expand_query("latest A1c?")
    assert "lvef" in expand_query("what was the ejection fraction")


def test_tokenizer_keeps_clinical_tokens():
    toks = tokenize("NT-proBNP was 6840 pg/mL; LVEF 30%")
    assert "nt-probnp" in toks and "6840" in toks and "lvef" in toks


def test_keyword_search_finds_exact_drug_name(retriever):
    res = retriever.search("P005", "oxaliplatin")
    assert res.max_bm25 > 0
    assert any("oxaliplatin" in c.text.lower() for c in res.chunks)
