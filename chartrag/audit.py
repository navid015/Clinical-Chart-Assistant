"""
Audit trail. Every request is logged as one JSON line with enough detail to reconstruct
what the system saw and did (retrieved chunks, citations, status, models, versions),
but with identifiers removed: the patient id is a salted hash and the question is redacted.
"""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone

from . import config

_PATTERNS = [
    (re.compile(r"\bSYN-\d{6}\b", re.I), "[MRN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
]


def redact(text: str, roster: list[dict]) -> str:
    for p in roster:
        for part in sorted({p["name"], *p["name"].split()}, key=len, reverse=True):
            if len(part) >= 3:
                text = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", text, flags=re.I)
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text


def patient_ref(patient_id: str) -> str:
    return hashlib.sha256(f"{config.AUDIT_SALT}:{patient_id}".encode()).hexdigest()[:12]


def _strip_pid(chunk_id: str) -> str:
    return chunk_id.split("/", 1)[1] if "/" in chunk_id else chunk_id


def log_request(result, patient_id: str, question: str, roster: list[dict]) -> str:
    request_id = result.request_id or uuid.uuid4().hex[:12]
    if not config.AUDIT_LOG:
        return request_id
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "patient_ref": patient_ref(patient_id),
        "question_redacted": redact(question, roster),
        "status": result.status,
        "reason": result.reason,
        "retrieved": [_strip_pid(c.id) for c in result.retrieved],
        "cited": [_strip_pid(c.id) for c in result.cited],
        "citation_coverage": result.citation_coverage,
        "verification": result.verification,
        "blocked_cross_patient": result.blocked_cross_patient,
        "latency_s": result.latency_s,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "pipeline_version": config.PIPELINE_VERSION,
    }
    with open(config.LOG_DIR / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return request_id
