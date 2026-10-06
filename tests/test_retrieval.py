import pytest

from src.rag.retrieve import Retriever, resolve_company


def test_known_query_returns_expected_passage(retriever, fixture_df):
    hits = retriever.retrieve("ransomware malware networks cyberattacks", k=5, company="XOM")
    cyber_ids = set(fixture_df[(fixture_df.ticker == "XOM") & (fixture_df.cyber == 1)].passage_id)
    assert hits[0].passage_id in cyber_ids


def test_filters_restrict_results(retriever):
    hits = retriever.retrieve("interest rates", k=8, sector="energy")
    assert hits and all(h.metadata["sector"] == "energy" for h in hits)
    hits = retriever.retrieve("interest rates", k=8, company="Bank of America")
    assert hits and all(h.metadata["ticker"] == "BAC" for h in hits)


def test_results_are_passage_level_and_sorted(retriever):
    hits = retriever.retrieve("supply chain vendors", k=8)
    ids = [h.passage_id for h in hits]
    assert len(ids) == len(set(ids))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_rerank_changes_scores(retriever):
    dense = retriever.retrieve("greenhouse gas emissions policy", k=3, rerank=False)
    reranked = retriever.retrieve("greenhouse gas emissions policy", k=3, rerank=True)
    assert all(h.score == h.dense_score for h in dense)
    assert any(h.score != h.dense_score for h in reranked)


def test_index_persists_across_clients(index_dir, retriever):
    again = Retriever(persist_dir=index_dir, backend="hashing")
    assert again.collection.count() == retriever.collection.count() > 0


def test_embedding_mismatch_fails_loudly(index_dir):
    with pytest.raises(RuntimeError, match="built with"):
        Retriever(persist_dir=index_dir, backend="hf")


def test_unknown_company_raises():
    with pytest.raises(KeyError):
        resolve_company("Tesla")


def test_eval_set_reports_unresolved_items(tmp_path, fixture_df):
    import json

    from src.rag.retrieve import load_eval_set

    passages = tmp_path / "p.parquet"
    fixture_df.to_parquet(passages)
    evalf = tmp_path / "e.jsonl"
    evalf.write_text(
        json.dumps(
            {
                "id": "ok",
                "question": "q1",
                "expected_rule": {"ticker": "XOM", "all_terms": ["greenhouse"]},
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "bad",
                "question": "q2",
                "expected_rule": {"ticker": "XOM", "all_terms": ["zzz-not-there"]},
            }
        )
        + "\n"
    )
    resolved, unresolved = load_eval_set(evalf, passages)
    assert [i["id"] for i in resolved] == ["ok"] and resolved[0]["expected_passage_ids"]
    assert [i["id"] for i in unresolved] == ["bad"]
