"""Dense retrieval with metadata filters and a lexical reranking pass, plus its evaluation.

Pipeline per query:
  1. dense top-N chunks from Chroma (N = 4k, cosine on MiniLM vectors), optionally
     filtered by ticker/company and/or sector;
  2. rerank: score = 0.7·dense + 0.3·coverage, where coverage is the idf-weighted share
     of the query's content words present in the chunk. Dense similarity finds the
     topic; coverage rewards chunks that actually mention the specific terms asked about;
  3. collapse chunks to passages (keep each passage's best chunk) and return top k.

Evaluation: hit rate@k and MRR@k for k in {1,3,5,8}, dense-only vs dense+rerank.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import FilterCondition, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_COLLECTION, CHROMA_DIR, EVAL_DIR, PASSAGES_PATH
from src.ingest.edgar import COMPANIES
from src.nlp.preprocess import content_terms
from src.rag.embeddings import backend_id, get_embed_model
from src.rag.index import chroma_client

DENSE_WEIGHT = 0.7
KS = [1, 3, 5, 8]
RETRIEVAL_EVAL_PATH = EVAL_DIR / "retrieval_eval.jsonl"


@dataclass
class Hit:
    passage_id: str
    text: str  # best-matching chunk text
    score: float
    dense_score: float
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "text": self.text,
            "company": self.metadata.get("company"),
            "ticker": self.metadata.get("ticker"),
            "sector": self.metadata.get("sector"),
            "filing_date": self.metadata.get("filing_date"),
            "categories": [c for c in self.metadata.get("categories", "").split(",") if c],
        }


def resolve_company(company: str | None) -> str | None:
    """Accept a ticker or (part of) a company name; return the ticker or raise."""
    if not company:
        return None
    key = company.strip().upper()
    if key in COMPANIES:
        return key
    for ticker, meta in COMPANIES.items():
        if company.strip().lower() in str(meta["name"]).lower():
            return ticker
    raise KeyError(f"Company {company!r} is not in the corpus")


class Retriever:
    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection: str = CHROMA_COLLECTION,
        backend: str | None = None,
    ) -> None:
        client = chroma_client(persist_dir)
        if collection not in [c.name for c in client.list_collections()]:
            raise FileNotFoundError(
                f"Chroma collection {collection!r} not found in {persist_dir}; "
                "run `python -m src.rag.index` first"
            )
        self.collection = client.get_collection(collection)
        built_with = (self.collection.metadata or {}).get("embed_model")
        if built_with != backend_id(backend):
            raise RuntimeError(
                f"Index was built with {built_with!r} but querying with "
                f"{backend_id(backend)!r}; rebuild the index or fix EMBED_BACKEND"
            )
        self.index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=self.collection),
            embed_model=get_embed_model(backend),
        )
        self._idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        docs = self.collection.get(include=["documents"])["documents"] or []
        n = len(docs)
        df: dict[str, int] = {}
        for d in docs:
            for tok in content_terms(d):
                df[tok] = df.get(tok, 0) + 1
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def coverage(self, query: str, text: str) -> float:
        q = content_terms(query)
        if not q:
            return 0.0
        present = content_terms(text)
        default = max(self._idf.values(), default=1.0)
        total = sum(self._idf.get(t, default) for t in q)
        return sum(self._idf.get(t, default) for t in q if t in present) / total

    def retrieve(
        self,
        query: str,
        k: int = 8,
        company: str | None = None,
        sector: str | None = None,
        rerank: bool = True,
    ) -> list[Hit]:
        filters = []
        ticker = resolve_company(company)
        if ticker:
            filters.append(MetadataFilter(key="ticker", value=ticker))
        if sector:
            filters.append(MetadataFilter(key="sector", value=sector.lower()))
        retriever = self.index.as_retriever(
            similarity_top_k=max(k * 4, 24),
            filters=MetadataFilters(filters=filters, condition=FilterCondition.AND)
            if filters
            else None,
        )
        best: dict[str, Hit] = {}
        for nws in retriever.retrieve(query):
            dense = float(nws.score or 0.0)
            text = nws.node.get_content()
            score = (
                DENSE_WEIGHT * dense + (1 - DENSE_WEIGHT) * self.coverage(query, text)
                if rerank
                else dense
            )
            pid = nws.node.metadata["passage_id"]
            if pid not in best or score > best[pid].score:
                best[pid] = Hit(pid, text, score, dense, dict(nws.node.metadata))
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]


def load_eval_set(
    path: Path = RETRIEVAL_EVAL_PATH, passages_path: Path = PASSAGES_PATH
) -> tuple[list[dict], list[dict]]:
    """Each item names its expected passages directly, or by a rule (ticker + required
    terms) resolved against passages.parquet.

    Returns (resolved, unresolved). Unresolved items are never silently dropped: every
    one is printed and recorded in RESULTS.md, and the question count reported is the
    resolved count, so a reader sees exactly what the metrics cover."""
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    passages = pd.read_parquet(passages_path) if passages_path.exists() else None
    resolved, unresolved = [], []
    for item in items:
        if item.get("expected_passage_ids"):
            resolved.append(item)
            continue
        rule = item.get("expected_rule")
        if rule is None or passages is None:
            raise ValueError(
                f"Eval item {item['id']} has no expected_passage_ids and no "
                "resolvable expected_rule"
            )
        mask = passages.ticker == rule["ticker"]
        for term in rule["all_terms"]:
            mask &= passages.text.str.contains(term, case=False, regex=True)
        ids = passages[mask].passage_id.tolist()
        if ids:
            item["expected_passage_ids"] = ids
            resolved.append(item)
        else:
            unresolved.append(item)
    for item in unresolved:
        print(
            f"WARNING eval item {item['id']} unresolved: rule {item['expected_rule']} "
            "matches no passage in the corpus; rewrite it"
        )
    if not resolved:
        raise ValueError("No retrieval eval item resolved against the corpus")
    return resolved, unresolved


def evaluate(retriever: Retriever, items: list[dict], rerank: bool) -> dict:
    hits = {k: 0 for k in KS}
    rr = {k: 0.0 for k in KS}
    for item in items:
        expected = set(item["expected_passage_ids"])
        ranked = [
            h.passage_id
            for h in retriever.retrieve(
                item["question"],
                k=max(KS),
                company=item.get("company"),
                sector=item.get("sector"),
                rerank=rerank,
            )
        ]
        first = next((i + 1 for i, pid in enumerate(ranked) if pid in expected), None)
        for k in KS:
            if first is not None and first <= k:
                hits[k] += 1
                rr[k] += 1.0 / first
    n = len(items)
    return {f"@{k}": {"hit_rate": round(hits[k] / n, 4), "mrr": round(rr[k] / n, 4)} for k in KS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--company")
    parser.add_argument("--sector")
    parser.add_argument("-k", type=int, default=8)
    args = parser.parse_args()
    retriever = Retriever()

    if args.query:
        for h in retriever.retrieve(args.query, k=args.k, company=args.company, sector=args.sector):
            print(f"{h.score:.3f} (dense {h.dense_score:.3f}) {h.passage_id}: {h.text[:160]}...")
    if args.eval:
        items, unresolved = load_eval_set()
        table = {}
        for name, rerank in [("dense_only", False), ("dense_plus_rerank", True)]:
            res = evaluate(retriever, items, rerank)
            table[name] = {f"{m}{k}": v[m] for k, v in res.items() for m in ["hit_rate", "mrr"]}
        cols = list(next(iter(table.values())))
        print(f"{'':<20}" + "".join(f"{c:>13}" for c in cols))
        for name, row in table.items():
            print(f"{name:<20}" + "".join(f"{row[c]:>13.3f}" for c in cols))
        from src.results import update

        update(
            "retrieval",
            {
                "n_questions": len(items),
                "unresolved_questions": [f"{i['id']}: {i['question']}" for i in unresolved]
                or "none",
                "metrics": table,
                "embed_model": backend_id(),
                "dense_weight": DENSE_WEIGHT,
            },
        )


if __name__ == "__main__":
    main()
