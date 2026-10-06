import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client(wired_tools):
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["passages_indexed"] > 0


def test_classify(client):
    r = client.post(
        "/classify",
        json={
            "text": "Ransomware and malware may disrupt our " "networks and expose customer data."
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["categories"]) == 6
    assert "cyber" in body["labels"]


def test_search(client):
    r = client.post(
        "/search", json={"query": "interest rates", "filters": {"sector": "banking"}, "k": 3}
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3 and all(p["sector"] == "banking" for p in results)
    assert (
        client.post(
            "/search", json={"query": "x risk", "filters": {"company": "Tesla"}}
        ).status_code
        == 404
    )


def test_ask_answer_and_refusal(client):
    r = client.post("/ask", json={"question": "What does JPMorgan say about credit losses?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and body["citations"]
    r = client.post("/ask", json={"question": "Will Microsoft's stock price rise?"})
    assert r.status_code == 200 and r.json()["refused"]


def test_docs_render(client):
    assert client.get("/docs").status_code == 200
    assert "/ask" in client.get("/openapi.json").json()["paths"]
