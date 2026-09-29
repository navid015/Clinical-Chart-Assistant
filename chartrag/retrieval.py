"""
Patient-scoped hybrid retrieval.

1. Clinical abbreviation expansion (A1c -> hemoglobin A1c, EF -> ejection fraction...).
2. Dense vector search, hard-filtered to the selected patient in the database query.
3. BM25 keyword search over the same patient's chunks (drug names, lab names and numbers
   are often matched better lexically than semantically).
4. Reciprocal rank fusion of the two rankings.
5. Defence in depth: any chunk whose patient_id differs from the request is dropped and
   counted. This should always be zero; the eval treats a non-zero count as a failure.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from . import config, llm
from .ingest import _client

ABBREVIATIONS = {
    "a1c": "hemoglobin a1c", "hba1c": "hemoglobin a1c", "egfr": "estimated glomerular filtration rate kidney function",
    "uacr": "urine albumin creatinine ratio albuminuria", "ldl": "ldl cholesterol", "bp": "blood pressure",
    "hr": "heart rate", "ef": "ejection fraction lvef", "lvef": "left ventricular ejection fraction",
    "hf": "heart failure", "hfref": "heart failure reduced ejection fraction", "chf": "congestive heart failure",
    "afib": "atrial fibrillation", "af": "atrial fibrillation", "cad": "coronary artery disease",
    "mi": "myocardial infarction", "icd": "implantable cardioverter defibrillator", "bnp": "nt-probnp natriuretic peptide",
    "copd": "chronic obstructive pulmonary disease", "osa": "obstructive sleep apnea", "cpap": "continuous positive airway pressure",
    "ahi": "apnea hypopnea index", "tsh": "thyroid stimulating hormone", "phq": "phq-9 depression score",
    "mdd": "major depressive disorder", "ckd": "chronic kidney disease", "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes", "htn": "hypertension", "gerd": "gastroesophageal reflux", "cea": "carcinoembryonic antigen",
    "capox": "capecitabine oxaliplatin chemotherapy", "ct": "computed tomography scan", "dexa": "bone density",
    "vte": "venous thromboembolism blood clot", "ed": "emergency department", "er": "emergency department",
    "nkda": "no known drug allergies", "sob": "shortness of breath", "fev1": "spirometry fev1",
    "hx": "history", "meds": "medications", "dx": "diagnosis", "rx": "prescription medication",
}

STOPWORDS = set("a an and are as at be by did do does for from has have he her him his how i in is it its "
                "me of on or she that the their them they this to was were what when which who why with "
                "you your patient patients any had ever been there has have most recent".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


def expand_query(query: str) -> str:
    """Add expansions for abbreviations in the query, and abbreviations for spelled-out terms
    (notes often say "LVEF" where a clinician asks about "ejection fraction", and vice versa)."""
    q = query.lower()
    extra = [ABBREVIATIONS[t] for t in re.findall(r"[a-z0-9]+", q) if t in ABBREVIATIONS]
    for abbr, full in ABBREVIATIONS.items():
        phrase = " ".join(full.split()[:2])
        if len(abbr) >= 2 and len(phrase) > 6 and phrase in q:
            extra.append(f"{abbr} {full}")
    return query if not extra else f"{query} ({'; '.join(dict.fromkeys(extra))})"


@dataclass
class RetrievedChunk:
    id: str
    patient_id: str
    text: str
    title: str
    doc_type: str
    date: str
    section: str
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0

    @property
    def body(self) -> str:
        return self.text.split("\n", 1)[1] if "\n" in self.text else self.text

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "title", "doc_type", "date", "section",
                                               "vector_score", "bm25_score", "fused_score")}


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]
    expanded_query: str
    max_similarity: float
    max_bm25: float
    blocked_cross_patient: int = 0
    candidates: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self, db_dir: Path | None = None):
        self.collection = _client(Path(db_dir or config.DB_DIR)).get_collection(config.COLLECTION)
        self._bm25 = {}

    def _patient_corpus(self, patient_id: str):
        if patient_id not in self._bm25:
            got = self.collection.get(where={"patient_id": patient_id}, include=["documents", "metadatas"])
            bm25 = BM25Okapi([tokenize(d) for d in got["documents"]]) if got["ids"] else None
            self._bm25[patient_id] = (bm25, got["ids"], got["documents"], got["metadatas"])
        return self._bm25[patient_id]

    def search(self, patient_id: str, query: str, k: int | None = None) -> SearchResult:
        k = k or config.TOP_K
        expanded = expand_query(query)
        bm25, ids, docs, metas = self._patient_corpus(patient_id)
        if not ids:
            return SearchResult([], expanded, 0.0, 0.0)
        pool = min(len(ids), k * 3)
        by_id = {}

        # Dense search, filtered to this patient inside the database query
        res = self.collection.query(query_embeddings=llm.embed([expanded]), n_results=pool,
                                    where={"patient_id": patient_id},
                                    include=["documents", "metadatas", "distances"])
        vector_rank = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            by_id[cid] = _make_chunk(cid, doc, meta)
            by_id[cid].vector_score = round(1.0 - dist, 4)
            vector_rank.append(cid)

        # Keyword search over the same patient's chunks
        scores = bm25.get_scores(tokenize(expanded))
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        bm25_rank = []
        for i in order[:pool]:
            if scores[i] <= 0:
                break
            cid = ids[i]
            by_id.setdefault(cid, _make_chunk(cid, docs[i], metas[i])).bm25_score = round(float(scores[i]), 4)
            bm25_rank.append(cid)

        # Reciprocal rank fusion
        fused = {}
        for ranking in (vector_rank, bm25_rank):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)

        # Defence in depth against cross-patient leakage
        blocked = 0
        chunks = []
        for cid in ranked:
            c = by_id[cid]
            if c.patient_id != patient_id:
                blocked += 1
                continue
            c.fused_score = round(fused[cid], 5)
            chunks.append(c)

        return SearchResult(
            chunks=chunks[:k],
            expanded_query=expanded,
            max_similarity=max((by_id[c].vector_score or 0.0 for c in vector_rank), default=0.0),
            max_bm25=max((by_id[c].bm25_score or 0.0 for c in bm25_rank), default=0.0),
            blocked_cross_patient=blocked,
            candidates=ranked,
        )


def _make_chunk(cid, doc, meta) -> RetrievedChunk:
    return RetrievedChunk(id=cid, patient_id=meta["patient_id"], text=doc, title=meta["title"],
                          doc_type=meta["doc_type"], date=meta.get("date", ""), section=meta["section"])
