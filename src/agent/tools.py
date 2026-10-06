"""The agent's three tools: plain typed functions, also exposed as LangChain tools.

Heavy resources (Chroma retriever, classifier) load lazily on first use and can be
swapped with `set_retriever` / `set_classifier` — the tests do this to run offline.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import StructuredTool

from src.config import CATEGORIES
from src.ingest.edgar import COMPANIES
from src.models.predict import Classifier, load_classifier, positive, to_categories
from src.rag.retrieve import Retriever, resolve_company

COMPANY_ALIASES: dict[str, list[str]] = {
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase", "chase"],
    "BAC": ["bank of america", "bofa"],
    "C": ["citigroup", "citi", "citibank"],
    "WFC": ["wells fargo"],
    "XOM": ["exxon", "exxonmobil", "exxon mobil", "mobil"],
    "CVX": ["chevron"],
    "COP": ["conocophillips", "conoco"],
    "SLB": ["schlumberger", "slb"],
    "MSFT": ["microsoft"],
    "AAPL": ["apple"],
    "NVDA": ["nvidia"],
    "ORCL": ["oracle"],
}
for _t, _m in COMPANIES.items():
    COMPANY_ALIASES[_t].append(str(_m["name"]).lower())

_state: dict[str, object] = {"retriever": None, "classifier": None}


def set_retriever(retriever: Retriever | None) -> None:
    _state["retriever"] = retriever
    corpus_info.cache_clear()


def set_classifier(classifier: Classifier | None) -> None:
    _state["classifier"] = classifier


def get_retriever() -> Retriever:
    if _state["retriever"] is None:
        _state["retriever"] = Retriever()
    return _state["retriever"]  # type: ignore[return-value]


def get_classifier() -> Classifier:
    if _state["classifier"] is None:
        _state["classifier"] = load_classifier()
    return _state["classifier"]  # type: ignore[return-value]


@lru_cache(maxsize=1)
def corpus_info() -> dict:
    """Which companies and filing years the index actually contains."""
    metas = get_retriever().collection.get(include=["metadatas"])["metadatas"] or []
    tickers = sorted({m["ticker"] for m in metas})
    years = sorted({int(str(m["filing_date"])[:4]) for m in metas})
    return {
        "tickers": tickers,
        "max_filing_year": years[-1] if years else None,
        "min_filing_year": years[0] if years else None,
        "aliases": {t: COMPANY_ALIASES.get(t, [t.lower()]) for t in tickers},
    }


def search_risk_disclosures(
    query: str, company: str | None = None, sector: str | None = None
) -> list[dict]:
    """Search 10-K risk-factor passages. Optionally filter by company (ticker or name)
    and/or sector ("banking", "energy", "technology"). Returns up to 8 passages with
    passage_id, score, text and metadata, best first."""
    return [
        h.as_dict() for h in get_retriever().retrieve(query, k=8, company=company, sector=sector)
    ]


def classify_passage(text: str) -> dict:
    """Classify a risk passage into credit, market, operational, regulatory, cyber and
    climate (multi-label). Returns per-category probabilities and the predicted labels."""
    clf = get_classifier()
    scores = to_categories(clf.predict_proba([text]))[0]
    return {"model": clf.name, "scores": scores, "labels": positive(scores)}


def compare_companies(company_a: str, company_b: str, category: str) -> dict:
    """Compare how two companies disclose one risk category: how many of each company's
    passages the classifier assigns to the category, and their most relevant passages."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
    retriever = get_retriever()
    out: dict = {"category": category, "companies": {}}
    for company in (company_a, company_b):
        ticker = resolve_company(company)
        metas = (
            retriever.collection.get(where={"ticker": ticker}, include=["metadatas"])["metadatas"]
            or []
        )
        all_pids = {m["passage_id"] for m in metas}
        flagged = {m["passage_id"] for m in metas if m.get(f"cat_{category}")}
        hits = retriever.retrieve(f"{category} risk", k=3, company=ticker)
        out["companies"][ticker] = {
            "company": str(COMPANIES[ticker]["name"]),
            "passages_total": len(all_pids),
            "passages_in_category": len(flagged),
            "share_in_category": round(len(flagged) / len(all_pids), 3) if all_pids else 0.0,
            "top_passages": [h.as_dict() for h in hits],
        }
    return out


def langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(f)
        for f in (search_risk_disclosures, classify_passage, compare_companies)
    ]


TOOLS = {f.__name__: f for f in (search_risk_disclosures, classify_passage, compare_companies)}
