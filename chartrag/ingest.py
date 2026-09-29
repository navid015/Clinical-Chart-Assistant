"""
Build the vector index.

Chunking is section-aware: each note section (Assessment, Plan, Findings...) becomes one
chunk, so a chunk never mixes, say, a plan with an unrelated exam finding. Structured data
(allergies, medications, problems, each lab test) become their own chunks. Every chunk
carries patient_id metadata, which retrieval uses as a hard filter.

Chunk ids are human-readable and stable, e.g.
    P001/2025-03-12_endocrinology-consult#plan
    P001/structured/labs/hemoglobin-a1c
which makes citations, audit logs and the golden test set easy to read.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings

from . import config, llm
from .records import Patient, load_patients


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def flag(value: float, ref: str) -> str:
    """Return H/L flag from a reference range like '4.0-5.6', '<30' or '>=60'."""
    ref = ref.replace(" ", "")
    try:
        if "-" in ref and not ref.startswith(("<", ">")):
            low, high = (float(x) for x in ref.split("-", 1))
            return "H" if value > high else "L" if value < low else ""
        if ref.startswith("<="):
            return "H" if value > float(ref[2:]) else ""
        if ref.startswith("<"):
            return "H" if value >= float(ref[1:]) else ""
        if ref.startswith(">="):
            return "L" if value < float(ref[2:]) else ""
        if ref.startswith(">"):
            return "L" if value <= float(ref[1:]) else ""
    except ValueError:
        pass
    return ""


def _chunk(pid, cid, title, doc_type, date, section, body, source):
    header = f"{title} ({doc_type}), {date}. Section: {section}."
    return {
        "id": f"{pid}/{cid}",
        "text": f"{header}\n{body}",
        "metadata": {"patient_id": pid, "title": title, "doc_type": doc_type,
                     "date": date, "section": section, "source": source},
    }


def structured_chunks(p: Patient) -> list[dict]:
    src = f"data/patients/{p.id}/patient.json"
    chunks = []

    lines = []
    for a in p.allergies:
        if a["type"] == "NKDA":
            lines.append(f"- No known drug allergies (recorded {a['recorded']}).")
        else:
            extra = f" Note: {a['note']}." if a.get("note") else ""
            lines.append(f"- {a['substance']}: {a['reaction']}; severity {a['severity'].lower()}; "
                         f"type {a['type'].lower()}; recorded {a['recorded']}.{extra}")
    latest = max((a["recorded"] for a in p.allergies), default="")
    chunks.append(_chunk(p.id, "structured/allergies", "Allergy and intolerance list", "Structured record",
                         latest, "Allergies", "Allergies and intolerances:\n" + "\n".join(lines), src))

    lines = [f"- {x['condition']} (ICD-10 {x['icd10']}); onset {x['onset']}; status {x['status'].lower()}."
             for x in p.problems]
    chunks.append(_chunk(p.id, "structured/problems", "Problem list", "Structured record",
                         "", "Problems", "Problem list:\n" + "\n".join(lines), src))

    def med_line(m):
        s = f"- {m['name']} {m['dose']} {m['route']}, {m['frequency']}; status {m['status'].lower()}; started {m['start']}"
        if m.get("stop"):
            s += f"; stopped {m['stop']}"
        if m.get("indication"):
            s += f"; for {m['indication']}"
        if m.get("reason"):
            s += f"; reason: {m['reason']}"
        if m.get("note"):
            s += f"; note: {m['note']}"
        return s + "."

    active = [med_line(m) for m in p.medications if m["status"] == "Active"]
    past = [med_line(m) for m in p.medications if m["status"] != "Active"]
    chunks.append(_chunk(p.id, "structured/medications-active", "Active medication list", "Structured record",
                         "", "Active medications", "Active medications:\n" + "\n".join(active), src))
    if past:
        chunks.append(_chunk(p.id, "structured/medications-history", "Medication history", "Structured record",
                             "", "Discontinued or completed medications",
                             "Discontinued or completed medications:\n" + "\n".join(past), src))

    for lab in p.labs:
        rows = []
        for r in lab["results"]:
            f = flag(r["value"], lab["ref_range"])
            rows.append(f"- {r['date']}: {r['value']} {lab['unit']}" + (f" ({f})" if f else ""))
        body = (f"{lab['test']} (LOINC {lab['loinc']}), reference range {lab['ref_range']} {lab['unit']}. "
                f"Results oldest to newest:\n" + "\n".join(rows))
        last = lab["results"][-1]["date"] if lab["results"] else ""
        chunks.append(_chunk(p.id, f"structured/labs/{slugify(lab['test'])}", f"Lab results: {lab['test']}",
                             "Structured record", last, "Lab results", body, src))
    return chunks


def note_chunks(p: Patient) -> list[dict]:
    chunks = []
    for n in p.notes:
        doc_slug = f"{n.date}_{slugify(n.title)}"
        for heading, text in n.sections:
            parts = _split_long(text)
            for i, part in enumerate(parts):
                suffix = f"-part{i + 1}" if len(parts) > 1 else ""
                chunks.append(_chunk(p.id, f"{doc_slug}#{slugify(heading)}{suffix}", n.title, n.doc_type,
                                     n.date, heading, part, n.path))
    return chunks


def _split_long(text: str, limit: int | None = None) -> list[str]:
    limit = limit or config.MAX_CHUNK_CHARS
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n"):
        if buf and len(buf) + len(para) + 1 > limit:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


def build_chunks(patients: dict[str, Patient]) -> list[dict]:
    chunks = []
    for p in patients.values():
        chunks.extend(structured_chunks(p))
        chunks.extend(note_chunks(p))
    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate chunk ids: {dupes}")
    return chunks


def data_fingerprint(data_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(data_dir).rglob("*")):
        if f.is_file():
            h.update(f.relative_to(data_dir).as_posix().encode())
            h.update(f.read_bytes())
    h.update(f"{config.EMBED_MODEL}|{config.LLM_MODE}|{config.CHUNKER_VERSION}".encode())
    return h.hexdigest()


def _client(db_dir: Path):
    return PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))


def build_index(db_dir: Path | None = None, data_dir: Path | None = None) -> int:
    db_dir = Path(db_dir or config.DB_DIR)
    data_dir = Path(data_dir or config.DATA_DIR)
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True)

    chunks = build_chunks(load_patients(data_dir))
    embeddings = llm.embed([c["text"] for c in chunks])

    collection = _client(db_dir).create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(ids=[c["id"] for c in chunks], embeddings=embeddings,
                   documents=[c["text"] for c in chunks], metadatas=[c["metadata"] for c in chunks])

    (db_dir / "manifest.json").write_text(json.dumps({
        "fingerprint": data_fingerprint(data_dir), "chunks": len(chunks),
        "embed_model": config.EMBED_MODEL, "llm_mode": config.LLM_MODE,
    }, indent=2))
    print(f"Indexed {len(chunks)} chunks from {len(load_patients(data_dir))} patients into {db_dir}")
    return len(chunks)


def ensure_index(db_dir: Path | None = None, data_dir: Path | None = None) -> None:
    """Rebuild only if the data, embedding model or chunker changed since the last build."""
    db_dir = Path(db_dir or config.DB_DIR)
    data_dir = Path(data_dir or config.DATA_DIR)
    manifest = db_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()).get("fingerprint") == data_fingerprint(data_dir):
            return
        print("Knowledge base changed; rebuilding index.")
    build_index(db_dir, data_dir)


if __name__ == "__main__":
    build_index()
