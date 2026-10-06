"""Isolate Item 1A "Risk Factors" from 10-K HTML and split it into passages.

EDGAR HTML is inconsistent (inline XBRL, tables used for layout, tables of contents
that repeat every heading). The strategy:

1. Flatten the document into "leaf block" paragraphs (block elements that contain no
   other block elements), so inline <span>s inside a paragraph stay together.
2. Find every candidate start ("Item 1A. Risk Factors") and end ("Item 1B",
   "Item 1C", "Item 2. Properties") heading. The table of contents produces a
   start/end pair with almost nothing between them, so pick the pair that encloses
   the most words.
3. Fallback: a standalone "Risk Factors" heading (some banks, e.g. JPM, put the
   section outside a literal "Item 1A" heading).
4. Pack paragraphs into 200-400 word passages on paragraph boundaries.

Every failure is logged with a reason and counted; nothing is dropped silently.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.config import EXTRACTION_LOG_PATH, PASSAGES_PATH, RAW_DIR

log = logging.getLogger(__name__)

BLOCK_TAGS = ["p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"]
MIN_WORDS, MAX_WORDS = 200, 400
MIN_SECTION_WORDS = 1500
THIN_PASSAGES = 15  # fewer than this from a 10-K usually means the wrong span was picked  # anything shorter is a TOC entry or a cross-reference

START_RE = re.compile(r"^\s*item\s*1a\s*[\.\:\-–—]?\s*(risk\s+factors)?\s*\.?\s*$", re.I)
START_INLINE_RE = re.compile(r"^\s*item\s*1a\s*[\.\:\-–—]?\s*risk\s+factors\b", re.I)
END_RE = re.compile(
    r"^\s*(item\s*1b\b|item\s*1c\b|item\s*2\s*[\.\:\-–—]?\s*(properties)?\s*\.?\s*$|"
    r"unresolved\s+staff\s+comments\s*\.?\s*$)",
    re.I,
)
FALLBACK_START_RE = re.compile(r"^\s*risk\s+factors\s*\.?\s*$", re.I)
NOISE_RE = re.compile(r"^(\d{1,3}|page \d+|table of contents|.{0,3})$", re.I)


class ExtractionError(Exception):
    """Raised with a short, countable reason when a filing cannot be extracted."""


@dataclass
class ExtractionResult:
    paragraphs: list[str]
    method: str


def html_to_paragraphs(html: str | bytes) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")
    paragraphs: list[str] = []
    for el in soup.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):
            continue  # not a leaf block; its children will be visited
        # Join inline elements with no separator, as a browser renders them. Filings that
        # style a first letter in its own <span> ("<span>R</span>isk Factors") otherwise
        # become "R isk Factors" and no heading matches (Oracle and SLB, pipeline run 6).
        text = re.sub(r"\s+", " ", el.get_text("")).strip()
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        if text and not NOISE_RE.match(text):
            paragraphs.append(text)
    return paragraphs


def _word_count(paragraphs: list[str]) -> int:
    return sum(len(p.split()) for p in paragraphs)


def isolate_item_1a(paragraphs: list[str]) -> ExtractionResult:
    starts = [i for i, p in enumerate(paragraphs) if START_RE.match(p) or START_INLINE_RE.match(p)]
    # "Item 1A." and "Risk Factors" are frequently split across two cells.
    for i, p in enumerate(paragraphs[:-1]):
        if re.fullmatch(r"\s*item\s*1a\s*[\.\:]?\s*", p, re.I) and FALLBACK_START_RE.match(
            paragraphs[i + 1]
        ):
            starts.append(i)
    ends = [i for i, p in enumerate(paragraphs) if END_RE.match(p)]

    best: tuple[int, int, int] | None = None
    for s in sorted(set(starts)):
        following = [e for e in ends if e > s]
        if not following:
            continue
        e = following[0]
        words = _word_count(paragraphs[s + 1 : e])
        if best is None or words > best[2]:
            best = (s, e, words)
    if best and best[2] >= MIN_SECTION_WORDS:
        s, e, _ = best
        return ExtractionResult(_strip_heading(paragraphs[s + 1 : e]), "item_1a_heading")

    fb_starts = [i for i, p in enumerate(paragraphs) if FALLBACK_START_RE.match(p)]
    structured = _risk_subsection_span(paragraphs, fb_starts)
    if structured and structured[2] >= MIN_SECTION_WORDS:
        s, e, _ = structured
        return ExtractionResult(paragraphs[s + 1 : e], "risk_subsections")

    best_fb: tuple[int, int, int] | None = None
    for s in fb_starts:
        following = [e for e in ends if e > s]
        e = following[0] if following else _next_section_heading(paragraphs, s)
        if e is None:
            continue
        words = _word_count(paragraphs[s + 1 : e])
        if best_fb is None or words > best_fb[2]:
            best_fb = (s, e, words)
    if best_fb and best_fb[2] >= MIN_SECTION_WORDS:
        s, e, _ = best_fb
        return ExtractionResult(_strip_heading(paragraphs[s + 1 : e]), "risk_factors_heading")

    if not starts and not fb_starts:
        raise ExtractionError("no_item_1a_heading")
    if not ends:
        raise ExtractionError("no_end_heading")
    raise ExtractionError("section_too_short")


def _is_caps_heading(p: str) -> bool:
    return p.isupper() and 1 <= len(p.split()) <= 8


def _risk_subsection_span(paragraphs: list[str], starts: list[int]) -> tuple[int, int, int] | None:
    """Annual-report layouts (Citi) have no "Item 1B/2" after the section. Their real
    "RISK FACTORS" heading is followed by ALL-CAPS subsections that all end in "RISKS"
    (STRATEGIC RISKS, CREDIT RISKS, ...); the section ends at the first ALL-CAPS heading
    that is not a risk subsection (e.g. SUSTAINABILITY). Summary and cross-reference
    mentions of "Risk Factors" lack that structure and are ignored."""
    best = None
    for s in starts:
        first_caps = next(
            (
                i
                for i in range(s + 1, min(s + 6, len(paragraphs)))
                if _is_caps_heading(paragraphs[i])
            ),
            None,
        )
        if first_caps is None or not paragraphs[first_caps].rstrip(" :").endswith("RISKS"):
            continue
        end = next(
            (
                i
                for i in range(first_caps + 1, len(paragraphs))
                if _is_caps_heading(paragraphs[i]) and "RISK" not in paragraphs[i]
            ),
            len(paragraphs),
        )
        words = _word_count(paragraphs[s + 1 : end])
        if best is None or words > best[2]:
            best = (s, end, words)
    return best


_NEXT_SECTION_RE = re.compile(
    r"^\s*(management'?s discussion and analysis|legal proceedings|properties|"
    r"quantitative and qualitative disclosures)\b.{0,40}$",
    re.I,
)


def _next_section_heading(paragraphs: list[str], start: int) -> int | None:
    for i in range(start + 1, len(paragraphs)):
        if _NEXT_SECTION_RE.match(paragraphs[i]) and _word_count(paragraphs[start + 1 : i]) > 0:
            return i
    return None


def _strip_heading(paragraphs: list[str]) -> list[str]:
    while paragraphs and FALLBACK_START_RE.match(paragraphs[0]):
        paragraphs = paragraphs[1:]
    return paragraphs


def _split_long(paragraph: str) -> list[str]:
    """Split a paragraph longer than MAX_WORDS on sentence boundaries."""
    sentences = re.split(r"(?<=[\.\?\!;])\s+", paragraph)
    chunks: list[str] = []
    current: list[str] = []
    for sent in sentences:
        words = sent.split()
        if len(words) > MAX_WORDS:  # pathological run-on sentence: hard split
            for j in range(0, len(words), MAX_WORDS):
                chunks.append(" ".join(words[j : j + MAX_WORDS]))
            continue
        if current and len(" ".join(current).split()) + len(words) > MAX_WORDS:
            chunks.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def pack_passages(paragraphs: list[str]) -> list[str]:
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p) if len(p.split()) > MAX_WORDS else [p])
    passages: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = len(unit.split())
        if current and current_words + n > MAX_WORDS:
            passages.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(unit)
        current_words += n
        if current_words >= MIN_WORDS:
            passages.append("\n\n".join(current))
            current, current_words = [], 0
    if current:
        tail = "\n\n".join(current)
        if passages and len(passages[-1].split()) + current_words <= MAX_WORDS:
            passages[-1] = passages[-1] + "\n\n" + tail
        else:
            passages.append(tail)  # short tail kept, not dropped; flagged via word_count
    return passages


HEADING_HINT_RE = re.compile(r"^\s*(item\s*1a|item\s*1b|item\s*1c|item\s*2\b|risk\s+factors)", re.I)


def heading_report(paragraphs: list[str], limit: int = 25) -> list[str]:
    """Every heading-like paragraph with the words that follow it until the next one:
    printed for failed or thin extractions so the log shows *why*."""
    hits = [i for i, p in enumerate(paragraphs) if HEADING_HINT_RE.match(p) and len(p) < 200]
    lines = []
    for n, i in enumerate(hits[:limit]):
        nxt = hits[n + 1] if n + 1 < len(hits) else len(paragraphs)
        lines.append(
            f"  [{i:>5}] {paragraphs[i][:90]!r} -> {_word_count(paragraphs[i + 1 : nxt])} words"
        )
    lines = lines or ["  (no heading-like paragraphs found)"]
    # Short ALL-CAPS lines after the first real "risk factors" heading: where a section
    # without an "Item 1B/2" end marker (e.g. Citi's annual-report layout) actually ends.
    starts = [i for i in hits if FALLBACK_START_RE.match(paragraphs[i])]
    if starts:
        start = max(starts, key=lambda i: _word_count(paragraphs[i + 1 : i + 400]))
        caps = [
            i
            for i in range(start + 1, len(paragraphs))
            if paragraphs[i].isupper() and 1 <= len(paragraphs[i].split()) <= 8
        ][:30]
        lines.append(f"  ALL-CAPS headings after [{start}]:")
        for n, i in enumerate(caps):
            nxt = caps[n + 1] if n + 1 < len(caps) else len(paragraphs)
            lines.append(
                f"    [{i:>5}] {paragraphs[i][:70]!r} -> {_word_count(paragraphs[i + 1 : nxt])} words"
            )
    return lines


def _try_isolate(path: Path) -> tuple[ExtractionResult | None, str, list[str]]:
    paragraphs = html_to_paragraphs(path.read_bytes())
    if not paragraphs:
        return None, "empty_document", []
    try:
        return isolate_item_1a(paragraphs), "", paragraphs
    except ExtractionError as exc:
        return None, str(exc), paragraphs


def extract_file(html_path: Path, meta: dict) -> tuple[list[dict], str]:
    result, reason, paragraphs = _try_isolate(html_path)
    ex13 = html_path.with_name(html_path.stem + ".ex13.htm")
    if ex13.exists():
        # Keep whichever document yields the longer risk-factor section.
        ex_result, _, _ = _try_isolate(ex13)
        if ex_result and (
            result is None or _word_count(ex_result.paragraphs) > _word_count(result.paragraphs)
        ):
            result = ExtractionResult(ex_result.paragraphs, "exhibit_13")
    if result is None:
        err = ExtractionError(reason)
        err.diagnostics = heading_report(paragraphs)
        if ex13.exists():
            err.diagnostics += [
                "  -- exhibit 13 --",
                *heading_report(html_to_paragraphs(ex13.read_bytes())),
            ]
        else:
            err.diagnostics.append("  (no exhibit 13 saved for this filing)")
        raise err
    passages = pack_passages(result.paragraphs)
    year = meta["filing_date"][:4]
    rows = [
        {
            "passage_id": f"{meta['ticker']}-{year}-{i:03d}",
            "cik": int(meta["cik"]),
            "ticker": meta["ticker"],
            "company": meta["company"],
            "sector": meta["sector"],
            "filing_date": meta["filing_date"],
            "text": text,
            "word_count": len(text.split()),
        }
        for i, text in enumerate(passages)
    ]
    return rows, result.method


def run_extraction(
    raw_dir: Path = RAW_DIR,
    out_path: Path = PASSAGES_PATH,
    log_path: Path = EXTRACTION_LOG_PATH,
    record_results: bool = True,
) -> pd.DataFrame:
    meta_files = sorted(raw_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No filings in {raw_dir}; run `python -m src.ingest.edgar` first")
    rows: list[dict] = []
    log_rows: list[dict] = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text())
        html_path = meta_path.with_suffix(".htm")
        entry = {
            "file": html_path.name,
            "ticker": meta["ticker"],
            "filing_date": meta["filing_date"],
        }
        try:
            file_rows, method = extract_file(html_path, meta)
        except ExtractionError as exc:
            log.warning("Extraction failed for %s: %s", html_path.name, exc)
            print(f"Heading candidates in {html_path.name}:")
            print("\n".join(getattr(exc, "diagnostics", [])))
            log_rows.append(
                {**entry, "status": "failed", "reason": str(exc), "method": "", "n_passages": 0}
            )
            continue
        if len(file_rows) < THIN_PASSAGES:
            print(f"Thin extraction ({len(file_rows)} passages, {method}) for {html_path.name}:")
            print("\n".join(heading_report(html_to_paragraphs(html_path.read_bytes()))))
        rows.extend(file_rows)
        log_rows.append(
            {**entry, "status": "ok", "reason": "", "method": method, "n_passages": len(file_rows)}
        )

    log_df = pd.DataFrame(log_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"All {len(log_df)} filings failed extraction; see {log_path}")
    df.to_parquet(out_path, index=False)

    failed = log_df[log_df.status == "failed"]
    summary = {
        "filings_total": len(log_df),
        "filings_extracted": int((log_df.status == "ok").sum()),
        "filings_failed": len(failed),
        "failure_rate": round(len(failed) / len(log_df), 4),
        "failure_reasons": failed.reason.value_counts().to_dict() or "none",
        "extraction_methods": log_df[log_df.status == "ok"].method.value_counts().to_dict(),
        "companies": int(df.ticker.nunique()),
        "sectors": int(df.sector.nunique()),
        "passages_total": len(df),
        "passages_per_sector": df.sector.value_counts().to_dict(),
        "passage_words_mean": round(float(df.word_count.mean()), 1),
        "passages_outside_200_400_words": int(
            ((df.word_count < MIN_WORDS) | (df.word_count > MAX_WORDS)).sum()
        ),
    }
    print(
        f"Extraction: {summary['filings_extracted']}/{summary['filings_total']} filings ok, "
        f"failure rate {summary['failure_rate']:.1%}, {len(df)} passages"
    )
    if record_results:
        from src.results import record

        record("ingestion", summary)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_extraction()
