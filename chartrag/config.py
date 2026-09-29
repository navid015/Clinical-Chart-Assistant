"""Central configuration. Every setting can be overridden with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "patients"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

# "openai" for real models, "fake" for offline tests and CI (deterministic, no API key needed)
LLM_MODE = os.getenv("LLM_MODE", "openai").lower()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Separate index per mode so fake and real embeddings never mix
DB_DIR = Path(os.getenv("VECTOR_DB_DIR", ROOT / f"vector_db_{LLM_MODE}"))
COLLECTION = "chart_chunks"

TOP_K = int(os.getenv("TOP_K", "6"))
# Evidence gate: below this cosine similarity (and with no keyword hit) we don't call the LLM.
# Tune with eval/run_eval.py; see docs/RISK_ANALYSIS.md (H-02).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.20"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
VERIFY_ANSWERS = os.getenv("VERIFY_ANSWERS", "true").lower() == "true"
AUDIT_LOG = os.getenv("AUDIT_LOG", "true").lower() == "true"
AUDIT_SALT = os.getenv("AUDIT_SALT", "change-me-in-production")

PIPELINE_VERSION = "1.0.0"
CHUNKER_VERSION = "1"
