"""Build the LlamaIndex → Chroma vector index over risk-factor passages.

Chunking: passages are 200-400 words, but all-MiniLM-L6-v2 truncates input at 256
word pieces, so embedding a whole passage would silently ignore its second half.
SentenceSplitter cuts each passage into ~200-token chunks (32-token overlap) on
sentence boundaries; every chunk keeps its parent `passage_id`, so retrieval is
scored and cited at passage level.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CATEGORIES,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHROMA_HOST,
    CHROMA_PORT,
    PASSAGES_PATH,
)
from src.models.predict import Classifier, load_classifier, positive, to_categories
from src.rag.embeddings import backend_id, get_embed_model

# chromadb 0.5.5's posthog client is incompatible with current posthog releases and logs
# an error on every call even with telemetry disabled; the message is noise.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 32
# Shown to the embedding model (company/sector context helps "Exxon climate" queries);
# ids, dates and predicted labels are filter/citation metadata only.
EMBED_EXCLUDED = [
    "passage_id",
    "ticker",
    "filing_date",
    "categories",
    *[f"cat_{c}" for c in CATEGORIES],
]


def chroma_client(persist_dir: Path = CHROMA_DIR):
    settings = Settings(anonymized_telemetry=False)
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=settings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir), settings=settings)


def to_documents(df: pd.DataFrame, category_scores: list[dict[str, float]]) -> list[Document]:
    docs = []
    for row, scores in zip(df.itertuples(index=False), category_scores, strict=True):
        cats = positive(scores)
        meta = {
            "passage_id": row.passage_id,
            "ticker": row.ticker,
            "company": row.company,
            "sector": row.sector,
            "filing_date": str(row.filing_date),
            "categories": ",".join(cats),
            **{f"cat_{c}": c in cats for c in CATEGORIES},  # Chroma metadata must be scalar
        }
        docs.append(
            Document(
                text=row.text,
                doc_id=row.passage_id,
                metadata=meta,
                excluded_embed_metadata_keys=EMBED_EXCLUDED,
                excluded_llm_metadata_keys=EMBED_EXCLUDED,
            )
        )
    return docs


def build_index(
    df: pd.DataFrame,
    classifier: Classifier | None = None,
    persist_dir: Path = CHROMA_DIR,
    collection: str = CHROMA_COLLECTION,
    backend: str | None = None,
) -> dict:
    classifier = classifier or load_classifier()
    t0 = time.perf_counter()
    scores = to_categories(classifier.predict_proba(df.text.tolist()))
    classify_s = time.perf_counter() - t0

    client = chroma_client(persist_dir)
    if collection in [c.name for c in client.list_collections()]:
        client.delete_collection(collection)  # full rebuild keeps the index consistent
    coll = client.create_collection(
        collection,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": backend_id(backend),
            "classifier": classifier.name,
        },
    )
    store = ChromaVectorStore(chroma_collection=coll)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(to_documents(df, scores))
    t0 = time.perf_counter()
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=get_embed_model(backend),
        show_progress=False,
    )
    embed_s = time.perf_counter() - t0
    return {
        "passages": len(df),
        "chunks": len(nodes),
        "collection": collection,
        "embed_model": backend_id(backend),
        "classifier": classifier.name,
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "classify_seconds": round(classify_s, 1),
        "embed_seconds": round(embed_s, 1),
        "stored_vectors": coll.count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    df = pd.read_parquet(PASSAGES_PATH)
    info = build_index(df)
    for k, v in info.items():
        print(f"{k}: {v}")
    from src.results import update

    update("retrieval", {"index": info})


if __name__ == "__main__":
    main()
