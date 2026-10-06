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
