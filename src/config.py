"""Central configuration: paths, seeds, model names, categories, environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb: no outbound telemetry

SEED = 42

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
RESULTS_JSON = ROOT / "results.json"
RESULTS_MD = ROOT / "RESULTS.md"

PASSAGES_PATH = PROCESSED_DIR / "passages.parquet"
EXTRACTION_LOG_PATH = PROCESSED_DIR / "extraction_log.csv"
LABELS_PATH = LABELS_DIR / "labels.csv"
SPLIT_PATH = LABELS_DIR / "split.json"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline" / "logreg.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert-risk"

CATEGORIES: list[str] = ["credit", "market", "operational", "regulatory", "cyber", "climate"]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_BASE = "distilbert-base-uncased"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / ".chroma")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "risk_passages")
CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf")  # "hf" or "hashing"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "")
SEC_REQUEST_DELAY_S = 0.15

AGENT_MAX_ITERATIONS = 5


def sec_user_agent() -> str:
    """SEC rejects requests without a descriptive User-Agent containing a contact email."""
    if (
        not SEC_CONTACT_EMAIL
        or "@" not in SEC_CONTACT_EMAIL
        or SEC_CONTACT_EMAIL.endswith("example.com")
    ):
        raise RuntimeError(
            "SEC_CONTACT_EMAIL is missing or a placeholder. Set a real contact email in .env "
            "(see .env.example); SEC blocks requests without one."
        )
    return f"risk-intelligence research project {SEC_CONTACT_EMAIL}"
