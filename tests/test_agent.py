import pytest

from src.agent import graph
from src.agent import guardrails as G

ALIASES = {"JPM": ["jpmorgan"], "XOM": ["exxon", "exxon mobil corp"], "AAPL": ["apple"]}


@pytest.mark.parametrize(
    "question,reason",
    [
        ("What are Tesla's main risk factors?", "company_not_in_corpus"),
        ("How does Goldman Sachs describe cyber risk?", "company_not_in_corpus"),
        ("Will Exxon's stock price go up next year?", "speculation"),
        ("Should I buy Apple shares?", "speculation"),
        ("What did JPMorgan disclose in 2031?", "after_filing_date"),
        ("What did Apple say on today's earnings call?", "after_filing_date"),
    ],
)
def test_scope_guardrail_refuses(question, reason):
    res = G.scope_check(question, ALIASES, max_filing_year=2024)
    assert not res.ok and res.reason == reason


def test_scope_guardrail_allows_in_scope():
    assert G.scope_check("What does JPMorgan say about cyber risk?", ALIASES, 2024).ok
    assert G.scope_check("Compare Exxon Mobil and Apple on climate risk", ALIASES, 2024).ok


def test_grounding_rejects_uncited_and_fabricated():
    passages = {"JPM-2024-001": "Cyberattacks could compromise the security of our systems."}
    ok = G.verify_grounding(
        'JPM states: "Cyberattacks could compromise the security of our '
        'systems." [JPM-2024-001]',
        passages,
    )
    assert ok.ok and ok.citations == ["JPM-2024-001"]
    assert not G.verify_grounding("JPM has strong cyber defenses.", passages).ok
    fabricated = G.verify_grounding(
        'JPM states: "Our systems have never been breached by ' 'anyone." [JPM-2024-001]', passages
    )
    assert not fabricated.ok
    assert not G.verify_grounding(
        'X: "Cyberattacks could compromise the security of our ' 'systems." [JPM-2024-999]',
        passages,
    ).ok


def test_agent_answers_with_citations(wired_tools):
    res = graph.ask("What does Exxon Mobil say about greenhouse gas emissions?")
    assert not res["refused"], res
    assert res["citations"] and all(c.startswith("XOM-") for c in res["citations"])
    for cid in res["citations"]:
        assert f"[{cid}]" in res["answer"]
    assert res["trace"][0].startswith("plan")
    assert any(t.startswith("act:") for t in res["trace"])


def test_agent_refuses_out_of_corpus(wired_tools):
    res = graph.ask("What are Tesla's battery supply risks?")
    assert res["refused"] and res["refusal_reason"].startswith("company_not_in_corpus")
    assert res["citations"] == []


def test_agent_refuses_when_grounding_fails(wired_tools, monkeypatch):
    monkeypatch.setattr(
        graph,
        "compose_rule_answer",
        lambda state: "Exxon will grow profits 20% next year. [XOM-2024-000]",
    )
    res = graph.ask("What does Exxon Mobil say about climate change regulation?")
    assert res["refused"] and res["refusal_reason"].startswith("ungrounded")


def test_agent_compare_and_classify(wired_tools):
    res = graph.ask("Compare Chevron and Exxon Mobil on climate risk")
    assert res["intent"] == "compare" and not res["refused"]
    assert {c.split("-")[0] for c in res["citations"]} <= {"CVX", "XOM"}
    res = graph.ask(
        'Classify this passage: "A data breach could expose confidential '
        'customer information and ransomware may disrupt our networks."'
    )
    assert res["intent"] == "classify" and "cyber" in res["answer"]


def test_iteration_cap(wired_tools, monkeypatch):
    monkeypatch.setattr(
        graph,
        "rule_plan",
        lambda q, info: ("search", [{"tool": "search_risk_disclosures", "args": {"query": q}}] * 9),
    )
    res = graph.ask("What does Apple say about supply chain disruptions?")
    assert res["iterations"] == 5


def test_agent_refuses_topic_absent_from_corpus(wired_tools):
    """Retrieval always returns *something*; the agent must not quote it if off-topic."""
    res = graph.ask("What does Apple say about deep-sea mining permits?")
    assert res["refused"], res["answer"]
    assert res["refusal_reason"].split(":")[0] in {"insufficient_evidence", "low_relevance"}


def test_quoted_sentences_are_on_topic(wired_tools):
    res = graph.ask("What interest rate risks do banks in the corpus describe?")
    assert not res["refused"]
    assert "tax" not in res["answer"].lower()
    assert "interest rate" in res["answer"].lower()
