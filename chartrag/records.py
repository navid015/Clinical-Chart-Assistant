"""Load synthetic patient records: structured data (patient.json) and clinical notes (notes/*.md)."""
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
import json

from . import config


@dataclass
class Note:
    patient_id: str
    doc_type: str
    title: str
    date: str
    author: str
    path: str
    sections: list[tuple[str, str]]  # (heading, text)


@dataclass
class Patient:
    data: dict
    notes: list[Note] = field(default_factory=list)

    id = property(lambda self: self.data["patient_id"])
    name = property(lambda self: self.data["name"])
    sex = property(lambda self: self.data["sex"])
    dob = property(lambda self: self.data["date_of_birth"])
    mrn = property(lambda self: self.data["mrn"])
    allergies = property(lambda self: self.data.get("allergies", []))
    problems = property(lambda self: self.data.get("problems", []))
    medications = property(lambda self: self.data.get("medications", []))
    labs = property(lambda self: self.data.get("labs", []))
    suggested_questions = property(lambda self: self.data.get("suggested_questions", []))

    def age(self, today: date | None = None) -> int:
        today = today or date.today()
        born = date.fromisoformat(self.dob)
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    @property
    def label(self) -> str:
        return f"{self.name} ({self.id}), {self.age()} {self.sex[0]}"

    @property
    def active_medications(self):
        return [m for m in self.medications if m.get("status") == "Active"]

    @property
    def active_problems(self):
        return [p for p in self.problems if p.get("status") == "Active"]


def parse_note(path: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} is missing front matter")
    header, body = text[4:].split("\n---\n", 1)
    meta = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    for required in ("patient_id", "doc_type", "title", "date"):
        if required not in meta:
            raise ValueError(f"{path} front matter is missing '{required}'")

    sections, heading, buf = [], "Summary", []
    for line in body.strip().splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    return Note(meta["patient_id"], meta["doc_type"], meta["title"], meta["date"],
                meta.get("author", ""), path.as_posix(), sections)


def load_patients(data_dir: Path | None = None) -> dict[str, Patient]:
    data_dir = Path(data_dir or config.DATA_DIR)
    patients = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        data = json.loads((folder / "patient.json").read_text(encoding="utf-8"))
        if data["patient_id"] != folder.name:
            raise ValueError(f"Folder {folder.name} does not match patient_id {data['patient_id']}")
        notes = [parse_note(f) for f in sorted((folder / "notes").glob("*.md"))]
        for n in notes:
            if n.patient_id != data["patient_id"]:
                raise ValueError(f"{n.path} belongs to {n.patient_id}, found in {folder.name}")
        patients[data["patient_id"]] = Patient(data, notes)
    return patients


@lru_cache(maxsize=1)
def get_patients() -> dict[str, Patient]:
    return load_patients()
