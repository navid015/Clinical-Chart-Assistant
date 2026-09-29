"""Deterministic safety checks that run before and after the language model."""
import re
from dataclasses import dataclass, field

ADVICE_PATTERNS = [
    r"\bshould (i|we|he|she|they|you)\b.{0,40}\b(start|stop|prescribe|give|increase|decrease|raise|lower|"
    r"change|switch|add|discontinue|hold|order|titrate|double)\b",
    r"\bwhat (dose|dosage|medication|drug)\b.{0,30}\bshould\b",
    r"\b(recommend|suggest)\b.{0,40}\b(treatment|medication|drug|therapy|dose|dosage|antibiotic)\b",
    r"\bwhat should (i|we) (do|give|prescribe|order)\b",
    r"\b(can you|please) (diagnose|prescribe)\b",
    r"\bhow (much|many mg)\b.{0,30}\b(give|take|prescribe)\b",
]

CITATION_RE = re.compile(r"\[S(\d+)\]")


def is_advice_request(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in ADVICE_PATTERNS)


def mentions_other_patient(question: str, current_id: str, roster: list[dict]) -> bool:
    """True if the question names, or gives the MRN or id of, a patient other than the selected one."""
    q = question.lower()
    words = set(re.findall(r"[a-z0-9\-]+", q))
    for p in roster:
        if p["id"] == current_id:
            continue
        name_parts = [w for w in re.findall(r"[a-z]+", p["name"].lower()) if len(w) >= 3]
        if any(part in words for part in name_parts):
            return True
        if p["mrn"].lower() in q or p["id"].lower() in words:
            return True
    return False


@dataclass
class CitationCheck:
    cited: set[int] = field(default_factory=set)
    invalid: set[int] = field(default_factory=set)
    uncited_sentences: list[str] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.cited) and not self.invalid


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    cited = {int(n) for n in CITATION_RE.findall(answer)}
    valid = set(range(1, n_sources + 1))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    factual = [s for s in sentences if len(s) >= 25 and not s.endswith(":")]
    uncited = [s for s in factual if not CITATION_RE.search(s)]
    coverage = 1.0 if not factual else (len(factual) - len(uncited)) / len(factual)
    return CitationCheck(cited & valid, cited - valid, uncited, round(coverage, 3))
