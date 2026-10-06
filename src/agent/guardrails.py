"""Scope checks before retrieval and grounding checks before answering.

The agent refuses rather than guesses. Three layers:

1. Scope (before any tool call): the question must be answerable from 10-K risk
   factors in this corpus. Refuse questions that
   - name a company that is not in the corpus,
   - ask about events after the latest filing date (or "today", "last week", ...),
   - ask for speculation: price targets, buy/sell advice, predictions.
2. Relevance (after retrieval): the best passage must clear MIN_RELEVANCE.
3. Grounding (after drafting): every sentence in the answer must cite at least one
   retrieved passage_id, and be supported by that passage — a quoted span must appear
   verbatim; an unquoted sentence needs >= MIN_SUPPORT of its content words in the cited
   text. Any unsupported sentence ⇒ refuse the whole answer.

Known failure mode: support is lexical. A sentence that reuses a passage's words but
flips its meaning ("is not exposed" vs "is exposed") passes. Quoting verbatim, which the
rule-based composer always does and the LLM prompt requires, is the mitigation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.nlp.preprocess import content_terms

MIN_RELEVANCE = 0.30
MIN_SUPPORT = 0.6
MIN_SENTENCE_RELEVANCE = 0.5
CITATION_RE = re.compile(r"\[([A-Z]{1,5}-\d{4}-\d{3})\]")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,})[\"”]")

SPECULATION_PATTERNS = [
    r"stock price",
    r"share price",
    r"price target",
    r"\bshould i (buy|sell|invest)",
    r"\b(buy|sell|short) (the )?(stock|shares)",
    r"\bwill .{0,40}\b(go up|go down|rise|fall|outperform|underperform|beat)\b",
    r"\bpredict",
    r"\bforecast",
    r"most likely to (fail|collapse|default|go bankrupt)",
    r"\binvest(ment)? advice",
]
TEMPORAL_PATTERNS = [
    r"\btoday'?s?\b",
    r"\byesterday\b",
    r"\blast (week|month)'?s?\b",
    r"\bthis (week|month)\b",
    r"\bright now\b",
    r"\bcurrently\b",
    r"\blatest news\b",
    r"\bearnings call\b",
    r"\bnext (quarter|year|month)\b",
]
# Capitalised words that are not company names.
NON_ENTITY_WORDS = {
    "what",
    "how",
    "why",
    "which",
    "who",
    "does",
    "do",
    "did",
    "is",
    "are",
    "will",
    "should",
    "can",
    "could",
    "would",
    "compare",
    "classify",
    "the",
    "a",
    "an",
    "in",
    "of",
    "on",
    "and",
    "or",
    "for",
    "to",
    "its",
    "their",
    "ceo",
    "cfo",
    "sec",
    "ai",
    "i",
    "item",
    "risk",
    "risks",
    "factors",
    "10-k",
    "us",
    "u.s",
    "fed",
    "federal",
    "reserve",
    "basel",
    "covid",
    "covid-19",
    "china",
    "taiwan",
    "europe",
    "european",
    "union",
    "eu",
    "uk",
    "united",
    "states",
    "america",
    "american",
    "russia",
    "ukraine",
    "paris",
    "agreement",
    "dodd-frank",
    "gdpr",
    "a passage",
    "according",
    "tell",
    "me",
    "list",
    "describe",
    "summarize",
    "give",
    "find",
    "show",
    "between",
    "versus",
    "vs",
}


@dataclass
class ScopeResult:
    ok: bool
    reason: str = ""
    message: str = ""


@dataclass
class GroundingResult:
    ok: bool
    citations: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    reason: str = ""


def _candidate_entities(question: str) -> list[str]:
    """Runs of capitalised words, e.g. 'Goldman Sachs', 'Tesla'."""
    spans = re.findall(r"\b([A-Z][A-Za-z&\.\-']+(?:\s+[A-Z][A-Za-z&\.\-']+)*)", question)
    out = []
    for span in spans:
        words = [re.sub(r"'s$", "", w).strip(".'") for w in span.split()]
        words = [w for w in words if w and w.lower() not in NON_ENTITY_WORDS]
        if words:
            out.append(" ".join(words))
    return out


def scope_check(
    question: str, company_aliases: dict[str, list[str]], max_filing_year: int | None
) -> ScopeResult:
    q = question.lower()
    for pat in SPECULATION_PATTERNS:
        if re.search(pat, q):
            return ScopeResult(
                False,
                "speculation",
                "This asks for a prediction or investment advice. Risk-factor "
                "disclosures describe what could go wrong; they do not support "
                "forecasts of prices or outcomes.",
            )
    for pat in TEMPORAL_PATTERNS:
        if re.search(pat, q):
            return ScopeResult(
                False,
                "after_filing_date",
                "This asks about recent or future events. The corpus only "
                "contains what companies disclosed in their 10-K filings"
                + (f" (latest filed in {max_filing_year})." if max_filing_year else "."),
            )
    if max_filing_year:
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
        if any(y > max_filing_year for y in years):
            return ScopeResult(
                False,
                "after_filing_date",
                f"The question refers to {max(years)}, but the latest filing in "
                f"the corpus is from {max_filing_year}.",
            )
    known = {a for aliases in company_aliases.values() for a in aliases}
    for ent in _candidate_entities(question):
        e = ent.lower()
        if any(e in k or k in e for k in known):
            continue
        return ScopeResult(
            False,
            "company_not_in_corpus",
            f"'{ent}' is not one of the companies in the corpus "
            f"({', '.join(sorted(company_aliases))}). I won't answer from "
            "general knowledge because I can't cite it.",
        )
    return ScopeResult(True)


def detect_companies(question: str, company_aliases: dict[str, list[str]]) -> list[str]:
    q = question.lower()
    found: list[tuple[int, str]] = []
    for ticker, aliases in company_aliases.items():
        positions = [m.start() for a in aliases for m in re.finditer(rf"\b{re.escape(a)}\b", q)]
        if len(ticker) > 1 and re.search(rf"\b{ticker}\b", question):
            positions.append(question.index(ticker))
        if positions:
            found.append((min(positions), ticker))
    return [t for _, t in sorted(found)]


def support(sentence: str, source: str) -> float:
    """Share of the sentence's content words that appear in the source text."""
    claim = content_terms(CITATION_RE.sub("", sentence))
    if not claim:
        return 1.0
    return len(claim & content_terms(source)) / len(claim)


def query_relevance(question: str, sentence: str, company_aliases: dict[str, list[str]]) -> float:
    """Share of the question's topic words (company names and filler removed) present in
    a sentence. 0 means the sentence shares nothing with what was asked."""
    q = question.lower()
    mentioned = [
        a
        for aliases in company_aliases.values()
        for a in aliases
        if re.search(rf"\b{re.escape(a)}\b", q)
    ]
    names = content_terms(" ".join(mentioned))
    topic = content_terms(question, drop=names)
    if not topic:
        return 0.0
    return len(topic & content_terms(sentence)) / len(topic)


def split_sentences(answer: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?\]])\s+(?=[A-Z\"“])", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def verify_grounding(answer: str, passages: dict[str, str]) -> GroundingResult:
    """passages: passage_id -> text that was actually retrieved during this run."""
    sentences = split_sentences(answer)
    if not sentences:
        return GroundingResult(False, reason="empty answer")
    cited_all: list[str] = []
    unsupported: list[str] = []
    for s in sentences:
        cited = CITATION_RE.findall(s)
        if not cited:
            unsupported.append(f"no citation: {s}")
            continue
        unknown = [c for c in cited if c not in passages]
        if unknown:
            unsupported.append(f"cites passages that were not retrieved {unknown}: {s}")
            continue
        source = " ".join(passages[c] for c in cited)
        norm_source = re.sub(r"\s+", " ", source)
        quotes = QUOTE_RE.findall(s)
        if quotes:
            missing = [q for q in quotes if re.sub(r"\s+", " ", q).strip(" .") not in norm_source]
            if missing:
                unsupported.append(f"quote not found in cited passage: {missing[0][:80]}")
                continue
        elif support(s, source) < MIN_SUPPORT:
            unsupported.append(f"support {support(s, source):.2f} < {MIN_SUPPORT}: {s}")
            continue
        cited_all.extend(c for c in cited if c not in cited_all)
    if unsupported:
        return GroundingResult(
            False,
            cited_all,
            unsupported,
            f"{len(unsupported)} of {len(sentences)} sentences not grounded",
        )
    return GroundingResult(True, cited_all)


def check_relevance(best_score: float | None) -> ScopeResult:
    if best_score is None:
        return ScopeResult(False, "no_passages", "No passage in the corpus matched the question.")
    if best_score < MIN_RELEVANCE:
        return ScopeResult(
            False,
            "low_relevance",
            f"The closest passage scored {best_score:.2f}, below the "
            f"{MIN_RELEVANCE:.2f} relevance floor, so I can't ground an answer.",
        )
    return ScopeResult(True)
