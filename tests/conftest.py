"""Tests run fully offline: fake deterministic model, temporary index and log folders."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="chartrag-test-")
os.environ["LLM_MODE"] = "fake"
os.environ["VECTOR_DB_DIR"] = os.path.join(_tmp, "db")
os.environ["LOG_DIR"] = os.path.join(_tmp, "logs")
os.environ["AUDIT_LOG"] = "true"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def patients():
    from chartrag.records import load_patients
    return load_patients()


@pytest.fixture(scope="session")
def index():
    from chartrag import config
    from chartrag.ingest import build_index
    build_index()
    return config.DB_DIR


@pytest.fixture(scope="session")
def retriever(index):
    from chartrag.retrieval import Retriever
    return Retriever(index)


@pytest.fixture(scope="session")
def assistant(retriever, patients):
    from chartrag.pipeline import ChartAssistant
    return ChartAssistant(retriever=retriever, patients=patients)


@pytest.fixture(scope="session")
def golden():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "eval" / "golden_set.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
