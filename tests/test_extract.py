from pathlib import Path

import pytest

from src.ingest.extract import (
    MAX_WORDS,
    MIN_WORDS,
    ExtractionError,
    extract_file,
    html_to_paragraphs,
    isolate_item_1a,
    pack_passages,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"
META = {
    "ticker": "TEST",
    "cik": 1,
    "company": "Synthetic Co",
    "sector": "technology",
    "filing_date": "2024-02-15",
}


def test_paragraphs_drop_hidden_and_noise():
    paras = html_to_paragraphs(FIXTURE.read_bytes())
    assert not any("hidden xbrl" in p for p in paras)
    assert "12" not in paras  # page numbers are noise


def test_isolates_real_section_not_table_of_contents():
    paras = html_to_paragraphs(FIXTURE.read_bytes())
    result = isolate_item_1a(paras)
    text = " ".join(result.paragraphs)
    assert result.method == "item_1a_heading"
    assert len(text.split()) > 1500
    assert "Unresolved Staff Comments" not in text
    assert "We lease offices" not in text
    assert "makes fixtures for tests" not in text


def test_passages_respect_word_bounds():
    paras = html_to_paragraphs(FIXTURE.read_bytes())
    passages = pack_passages(isolate_item_1a(paras).paragraphs)
    counts = [len(p.split()) for p in passages]
    assert all(c <= MAX_WORDS for c in counts)
    assert all(c >= MIN_WORDS for c in counts[:-1])  # only the tail may be short


def test_extract_file_schema(tmp_path):
    rows, method = extract_file(FIXTURE, META)
    assert rows and method == "item_1a_heading"
    assert set(rows[0]) >= {"passage_id", "cik", "company", "sector", "filing_date", "text"}
    assert rows[0]["passage_id"] == "TEST-2024-000"


def test_missing_section_fails_loudly(tmp_path):
    with pytest.raises(ExtractionError, match="no_item_1a_heading"):
        isolate_item_1a(["Item 1. Business", "We make things."])


THIN_10K = """<html><body>
<p>Item 1A. Risk Factors</p>
<p>Information in response to this item can be found in the Risk Factors section of the
Annual Report, which is incorporated by reference.</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p></body></html>"""


def test_falls_back_to_exhibit_13(tmp_path):
    primary = tmp_path / "WFC_2025-02-25_x.htm"
    primary.write_text(THIN_10K)
    primary.with_name(primary.stem + ".ex13.htm").write_bytes(FIXTURE.read_bytes())
    rows, method = extract_file(primary, {**META, "ticker": "WFC"})
    assert method == "exhibit_13" and len(rows) >= 5


def test_failure_carries_heading_diagnostics(tmp_path):
    primary = tmp_path / "WFC_2025-02-25_x.htm"
    primary.write_text(THIN_10K)
    with pytest.raises(ExtractionError, match="section_too_short") as info:
        extract_file(primary, {**META, "ticker": "WFC"})
    report = "\n".join(info.value.diagnostics)
    assert "Item 1A. Risk Factors" in report and "Item 1B" in report


def test_inline_spans_do_not_split_words():
    """Filings that style a first letter separately must still yield 'Risk Factors'."""
    html = (
        '<p><span style="font-size:14pt">I</span>tem 1A. <span>R</span>isk Factors</p>'
        "<p>Line one<br>line two</p>"
    )
    paras = html_to_paragraphs(html)
    assert paras[0] == "Item 1A. Risk Factors"
    assert paras[1] == "Line one line two"


CITI_LIKE = "".join(
    [
        "<p>Risk Factors</p><p>See the Risk Factors section for a discussion.</p>",  # summary mention
        "<p>MANAGEMENT'S DISCUSSION</p>" + "<p>" + "Revenue grew in the period. " * 300 + "</p>",
        "<p>RISK FACTORS</p>",
        "<p>STRATEGIC RISKS</p>",
        "<p>" + "Changes in strategy could adversely affect results. " * 120 + "</p>",
        "<p>CREDIT RISKS</p>",
        "<p>" + "Borrowers may default on loans and counterparties may fail. " * 120 + "</p>",
        "<p>SUSTAINABILITY</p>",
        "<p>" + "We publish an annual sustainability report. " * 200 + "</p>",
    ]
)


def test_risk_subsection_layout_ends_at_first_non_risk_heading():
    result = isolate_item_1a(html_to_paragraphs(CITI_LIKE))
    text = " ".join(result.paragraphs)
    assert result.method == "risk_subsections"
    assert "Borrowers may default" in text and "Changes in strategy" in text
    assert "sustainability report" not in text and "Revenue grew" not in text
