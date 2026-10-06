"""Embedding model factory.

"hf"      → sentence-transformers/all-MiniLM-L6-v2 via LlamaIndex's HuggingFaceEmbedding:
            384-dim, mean-pooled, L2-normalised sentence vectors; 22M parameters, fast on CPU.
"hashing" → a deterministic bag-of-words hashing embedding (scikit-learn). No download, no
            semantics beyond word overlap. Exists so tests/CI run offline; never use it for
            reported retrieval metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr
from sklearn.feature_extraction.text import HashingVectorizer

from src.config import EMBED_BACKEND, EMBED_MODEL, MODELS_DIR
from src.nlp.preprocess import normalize, tokenize

EMBED_DIM = 384


class HashingEmbedding(BaseEmbedding):
    _vectorizer: HashingVectorizer = PrivateAttr()

    def __init__(self, dim: int = EMBED_DIM, **kwargs: Any) -> None:
        super().__init__(model_name=f"hashing-{dim}", **kwargs)
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            preprocessor=normalize,
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 1),
            alternate_sign=False,
            norm="l2",
        )

    def _embed(self, text: str) -> list[float]:
        vec = self._vectorizer.transform([text]).toarray()[0]
        if not np.any(vec):  # all-stopword text: avoid a zero vector (undefined cosine)
            vec = np.full_like(vec, 1.0 / np.sqrt(len(vec)))
        return vec.astype(float).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)


def get_embed_model(backend: str | None = None) -> BaseEmbedding:
    backend = backend or EMBED_BACKEND
    if backend == "hashing":
        return HashingEmbedding()
    if backend == "hf":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(
            model_name=EMBED_MODEL, device="cpu", cache_folder=str(MODELS_DIR / "hf-cache")
        )
    raise ValueError(f"Unknown EMBED_BACKEND {backend!r}; use 'hf' or 'hashing'")


def backend_id(backend: str | None = None) -> str:
    backend = backend or EMBED_BACKEND
    return EMBED_MODEL if backend == "hf" else f"hashing-{EMBED_DIM}"
