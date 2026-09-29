"""Post-generation faithfulness check: a second model call verifies each claim against the sources."""
from dataclasses import dataclass, field

from pydantic import BaseModel

from . import config, llm
from .prompts import VERIFY_SYSTEM


class Claim(BaseModel):
    claim: str
    supported: bool
    sources: list[str] = []


class Verification(BaseModel):
    claims: list[Claim]


@dataclass
class VerificationResult:
    total: int = 0
    supported: int = 0
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def score(self) -> float | None:
        return round(self.supported / self.total, 3) if self.total else None

    def to_dict(self) -> dict:
        return {"total": self.total, "supported": self.supported, "score": self.score,
                "unsupported": self.unsupported, "error": self.error}


def format_sources(chunks) -> str:
    return "\n\n".join(f"[S{i}] {c.title} ({c.doc_type}), {c.date or 'undated'}, section: {c.section}\n{c.body}"
                       for i, c in enumerate(chunks, 1))


def verify_answer(answer: str, chunks, model: str | None = None) -> VerificationResult:
    messages = [{"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": f"Sources:\n{format_sources(chunks)}\n\nAnswer:\n{answer}"}]
    try:
        v = llm.chat_json(messages, Verification, model=model or config.JUDGE_MODEL)
    except Exception as exc:  # verification failure is reported, never silently treated as a pass
        return VerificationResult(error=f"{type(exc).__name__}: {exc}"[:200])
    return VerificationResult(total=len(v.claims), supported=sum(c.supported for c in v.claims),
                              unsupported=[c.claim for c in v.claims if not c.supported])
