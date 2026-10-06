"""EDGAR client logic with a fake HTTP session: no network, no SEC requests."""

import pytest

from src.ingest import edgar


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.content = b""

    def json(self):
        return self._payload


def cols(rows):
    return {
        "form": [r[0] for r in rows],
        "accessionNumber": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "primaryDocument": [r[3] for r in rows],
    }


class FakeSession:
    """A bank-like filer: the recent window holds one 10-K among many other forms."""

    def __init__(self):
        self.headers = {}
        self.urls = []
        recent = [("424B2", f"a{i}", "2026-01-01", "x.htm") for i in range(5)]
        recent.insert(2, ("10-K", "k2026", "2026-02-13", "jpm-2025.htm"))
        self.pages = {
            edgar.SUBMISSIONS_URL.format(cik=19617): {
                "filings": {
                    "recent": cols(recent),
                    "files": [{"name": "CIK0000019617-submissions-001.json"}],
                }
            },
            edgar.SUBMISSIONS_PAGE_URL.format(name="CIK0000019617-submissions-001.json"): cols(
                [
                    ("8-K", "e1", "2025-06-01", "y.htm"),
                    ("10-K", "k2025", "2025-02-14", "jpm-2024.htm"),
                    ("10-K", "k2024", "2024-02-16", "jpm-2023.htm"),
                    ("10-K", "k2023", "2023-02-21", "jpm-2022.htm"),
                ]
            ),
        }

    def get(self, url, timeout):
        self.urls.append(url)
        if url.endswith("index.json"):
            items = [{"name": "jpm-2025.htm"}, {"name": "jpm-ex13.htm"}, {"name": "R1.htm"}]
            return FakeResponse({"directory": {"item": items}})
        if url.endswith(".htm"):
            resp = FakeResponse({})
            resp.content = f"<html>{url}</html>".encode()
            return resp
        return FakeResponse(self.pages[url])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(edgar, "sec_user_agent", lambda: "test agent test@example.org")
    monkeypatch.setattr(edgar, "SEC_REQUEST_DELAY_S", 0)
    return edgar.EdgarClient(session=FakeSession())


def test_reads_older_pages_when_recent_window_is_short(client):
    refs = client.recent_10k("JPM", 3)
    assert [r.accession for r in refs] == ["k2026", "k2025", "k2024"]
    assert refs[0].url.endswith("/19617/k2026/jpm-2025.htm")


def test_stops_paging_once_enough_found(client):
    refs = client.recent_10k("JPM", 1)
    assert [r.accession for r in refs] == ["k2026"]
    assert len(client.session.urls) == 1  # never fetched the older page


def test_download_saves_exhibit_13_when_present(client, monkeypatch, tmp_path):
    monkeypatch.setattr(edgar, "RAW_DIR", tmp_path)
    ref = client.recent_10k("JPM", 1)[0]
    path, cached = client.download(ref)
    assert not cached and path.endswith(".htm")
    ex13 = tmp_path / f"{ref.raw_path_stem}.ex13.htm"
    assert ex13.exists() and b"jpm-ex13.htm" in ex13.read_bytes()
    assert (tmp_path / f"{ref.raw_path_stem}.json").exists()
