"""One classifier interface for the agent and the API.

Prefers the fine-tuned DistilBERT; falls back to the TF-IDF logistic regression so the
system (and CI, which never downloads models) still runs before Phase 3 has happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from src.config import BASELINE_MODEL_PATH, CATEGORIES, DISTILBERT_DIR

THRESHOLD = 0.5


class Classifier(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


class BaselineClassifier:
    name = "tfidf_logreg"

    def __init__(self, path: Path = BASELINE_MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No baseline model at {path}; run `python -m src.nlp.baseline`")
        self.pipe = joblib.load(path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.pipe.predict_proba(texts))


class _DistilBert:
    name = "distilbert"

    def __init__(self) -> None:
        from src.models.finetune import DistilBertClassifier

        self.inner = DistilBertClassifier()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.inner.predict_proba(texts)


def load_classifier() -> Classifier:
    if (DISTILBERT_DIR / "config.json").exists():
        return _DistilBert()
    return BaselineClassifier()


def to_categories(probs: np.ndarray) -> list[dict[str, float]]:
    return [{c: round(float(p), 4) for c, p in zip(CATEGORIES, row, strict=True)} for row in probs]


def positive(scores: dict[str, float], threshold: float = THRESHOLD) -> list[str]:
    return [c for c, p in scores.items() if p >= threshold]
