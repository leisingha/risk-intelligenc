"""Fetch recent 10-K filings from SEC EDGAR.

Honours SEC's fair-access rules: a descriptive User-Agent with a contact email on
every request, and a 150ms delay between requests (well under 10 req/s).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass

import requests

from src.config import RAW_DIR, SEC_REQUEST_DELAY_S, sec_user_agent

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"

# 12 companies across 3 sectors. 3 filings each gives 36 filings (spec: 30-40).
COMPANIES: dict[str, dict[str, str | int]] = {
    "JPM": {"cik": 19617, "name": "JPMorgan Chase & Co.", "sector": "banking"},
    "BAC": {"cik": 70858, "name": "Bank of America Corp", "sector": "banking"},
    "C": {"cik": 831001, "name": "Citigroup Inc", "sector": "banking"},
    "WFC": {"cik": 72971, "name": "Wells Fargo & Co", "sector": "banking"},
    "XOM": {"cik": 34088, "name": "Exxon Mobil Corp", "sector": "energy"},
    "CVX": {"cik": 93410, "name": "Chevron Corp", "sector": "energy"},
    "COP": {"cik": 1163165, "name": "ConocoPhillips", "sector": "energy"},
    "SLB": {"cik": 87347, "name": "Schlumberger Ltd", "sector": "energy"},
    "MSFT": {"cik": 789019, "name": "Microsoft Corp", "sector": "technology"},
    "AAPL": {"cik": 320193, "name": "Apple Inc", "sector": "technology"},
    "NVDA": {"cik": 1045810, "name": "NVIDIA Corp", "sector": "technology"},
    "ORCL": {"cik": 1341439, "name": "Oracle Corp", "sector": "technology"},
}


@dataclass
class FilingRef:
    ticker: str
    cik: int
    company: str
    sector: str
    accession: str
    filing_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.primary_document
        )

    @property
    def raw_path_stem(self) -> str:
        return f"{self.ticker}_{self.filing_date}_{self.accession}"


class EdgarClient:
    def __init__(self, session: requests.Session | None = None, max_retries: int = 3) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": sec_user_agent(), "Accept-Encoding": "gzip, deflate"}
        )
        self.max_retries = max_retries
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = SEC_REQUEST_DELAY_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 403:
                    raise RuntimeError(
                        f"SEC returned 403 for {url}. Usually a missing/invalid User-Agent "
                        "contact email, or the network blocks sec.gov."
                    )
                last_error = RuntimeError(f"HTTP {resp.status_code} for {url}")
            except requests.RequestException as exc:
                last_error = exc
            backoff = 2 ** (attempt - 1)
            log.warning(
                "Attempt %d/%d failed for %s: %s; retrying in %ss",
                attempt,
                self.max_retries,
                url,
                last_error,
                backoff,
            )
            time.sleep(backoff)
        raise RuntimeError(f"Giving up on {url} after {self.max_retries} attempts") from last_error

    def recent_10k(self, ticker: str, limit: int) -> list[FilingRef]:
        """Most recent `limit` 10-Ks. EDGAR's `filings.recent` only holds the latest ~1000
        filings; large banks file so many other forms that it can contain a single 10-K
        (pipeline run 5 got one each for JPM, BAC and C). Older filings live in the
        paginated files listed under `filings.files`, which are read until enough 10-Ks
        are found."""
        meta = COMPANIES[ticker]
        data = self.get(SUBMISSIONS_URL.format(cik=int(meta["cik"]))).json()
        refs = self._collect_10k(ticker, data["filings"]["recent"], limit)
        for page in data["filings"].get("files", []):
            if len(refs) >= limit:
                break
            older = self.get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json()
            refs += self._collect_10k(ticker, older, limit - len(refs))
        return refs

    @staticmethod
    def _collect_10k(ticker: str, columns: dict, limit: int) -> list[FilingRef]:
        meta = COMPANIES[ticker]
        refs: list[FilingRef] = []
        for form, acc, date, doc in zip(
            columns["form"],
            columns["accessionNumber"],
            columns["filingDate"],
            columns["primaryDocument"],
            strict=True,
        ):
            if form != "10-K":
                continue
            refs.append(
                FilingRef(
                    ticker, int(meta["cik"]), str(meta["name"]), str(meta["sector"]), acc, date, doc
                )
            )
            if len(refs) >= limit:
                break
        return refs

    def download(self, ref: FilingRef) -> tuple[str, bool]:
        """Save the filing HTML and its metadata. Returns (path, was_cached)."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        html_path = RAW_DIR / f"{ref.raw_path_stem}.htm"
        meta_path = RAW_DIR / f"{ref.raw_path_stem}.json"
        if html_path.exists() and meta_path.exists():
            return str(html_path), True
        resp = self.get(ref.url)
        html_path.write_bytes(resp.content)
        self._download_exhibit_13(ref)
        meta_path.write_text(json.dumps({**asdict(ref), "url": ref.url}, indent=2))
        return str(html_path), False

    def _download_exhibit_13(self, ref: FilingRef) -> None:
        """Some banks (e.g. Wells Fargo) file a thin 10-K whose Item 1A only points to the
        "Risk Factors" section of the Annual Report, filed as Exhibit 13. Save it so
        extraction can fall back to it."""
        listing = self.get(INDEX_URL.format(cik=ref.cik, accession=ref.accession.replace("-", "")))
        names = [i["name"] for i in listing.json()["directory"]["item"]]
        ex13 = [n for n in names if re.search(r"ex-?13", n, re.I) and n.lower().endswith(".htm")]
        if ex13:
            resp = self.get(
                ARCHIVE_URL.format(
                    cik=ref.cik, accession=ref.accession.replace("-", ""), document=ex13[0]
                )
            )
            (RAW_DIR / f"{ref.raw_path_stem}.ex13.htm").write_bytes(resp.content)


def fetch(tickers: list[str], per_company: int) -> list[FilingRef]:
    unknown = [t for t in tickers if t not in COMPANIES]
    if unknown:
        raise ValueError(f"Unknown tickers {unknown}; known: {sorted(COMPANIES)}")
    client = EdgarClient()
    fetched: list[FilingRef] = []
    for ticker in tickers:
        refs = client.recent_10k(ticker, per_company)
        if not refs:
            log.error("No 10-K filings found for %s", ticker)
        for ref in refs:
            path, cached = client.download(ref)
            log.info(
                "%s %s %s -> %s", "cached" if cached else "fetched", ticker, ref.filing_date, path
            )
            fetched.append(ref)
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 10-K filings from EDGAR, then extract.")
    parser.add_argument(
        "--tickers", default=",".join(COMPANIES), help="Comma-separated tickers (default: all 12)"
    )
    parser.add_argument("--per-company", type=int, default=3)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only fetch one filing index and print the first 10-K's metadata",
    )
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.smoke:
        refs = EdgarClient().recent_10k(tickers[0], 1)
        print(json.dumps({**asdict(refs[0]), "url": refs[0].url}, indent=2))
        return

    refs = fetch(tickers, args.per_company)
    print(f"Fetched {len(refs)} filings for {len(tickers)} companies")
    if not args.no_extract:
        from src.ingest.extract import run_extraction

        df = run_extraction()
        print(f"passages.parquet rows: {len(df)}")
        for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
            print(f"\n--- {row.passage_id} ({row.company}, {row.filing_date}) ---")
            print(row.text[:600] + ("..." if len(row.text) > 600 else ""))


if __name__ == "__main__":
    main()
