"""FastAPI service: /health, /classify, /search, /ask. OpenAPI docs at /docs."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import tools
from src.config import CATEGORIES

app = FastAPI(
    title="Risk Intelligence API",
    description="Classify, search and ask questions about 10-K Item 1A risk factors. "
    "Answers cite passage_ids or refuse.",
    version="1.0.0",
)


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        examples=[
            "A breach of our information systems could expose customer data and lead to fines."
        ],
    )


class CategoryScore(BaseModel):
    category: str
    confidence: float
    predicted: bool


class ClassifyResponse(BaseModel):
    model: str
    labels: list[str]
    categories: list[CategoryScore]


class SearchFilters(BaseModel):
    company: str | None = Field(None, description="Ticker or company name, e.g. 'XOM'")
    sector: Literal["banking", "energy", "technology"] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["climate change regulation"])
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(8, ge=1, le=20)


class Passage(BaseModel):
    passage_id: str
    score: float
    dense_score: float
    company: str | None
    ticker: str | None
    sector: str | None
    filing_date: str | None
    categories: list[str]
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[Passage]


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, examples=["What does JPMorgan say about cybersecurity risk?"]
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str
    citations: list[str]
    planner: str | None
    iterations: int
    trace: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    passages_indexed: int | None
    classifier: str | None
    detail: str = ""


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detail = []
    try:
        retriever = tools.get_retriever()
        n = retriever.collection.count()
        index_ok = True
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        n, index_ok = None, False
        detail.append(f"index: {exc}")
    try:
        clf_name = tools.get_classifier().name
    except (FileNotFoundError, OSError) as exc:
        clf_name = None
        detail.append(f"classifier: {exc}")
    status = "ok" if index_ok and clf_name else "degraded"
    return HealthResponse(
        status=status,
        index_loaded=index_ok,
        passages_indexed=n,
        classifier=clf_name,
        detail="; ".join(detail),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    res = tools.classify_passage(req.text)
    return ClassifyResponse(
        model=res["model"],
        labels=res["labels"],
        categories=[
            CategoryScore(category=c, confidence=res["scores"][c], predicted=c in res["labels"])
            for c in CATEGORIES
        ],
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = tools.get_retriever().retrieve(
            req.query, k=req.k, company=req.filters.company, sector=req.filters.sector
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    return SearchResponse(query=req.query, results=[Passage(**h.as_dict()) for h in hits])


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    from src.agent.graph import ask as agent_ask

    res = agent_ask(req.question)
    return AskResponse(
        question=res["question"],
        answer=res["answer"],
        refused=res["refused"],
        refusal_reason=res["refusal_reason"] or "",
        citations=res["citations"] or [],
        planner=res["planner"],
        iterations=res["iterations"] or 0,
        trace=res["trace"] or [],
    )
