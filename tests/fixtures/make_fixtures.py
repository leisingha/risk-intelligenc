"""Generate the SYNTHETIC test fixtures (deterministic, seed 42).

These are not real SEC filings. The sentences are generic, invented risk-factor
language so tests and CI can run offline without EDGAR or model downloads. Real
tickers are used only so the fixtures exercise the same company-resolution code as
production; no sentence here is a quote from any company's filing.

    python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
CATEGORIES = ["credit", "market", "operational", "regulatory", "cyber", "climate"]

TEMPLATES = {
    "credit": [
        "Borrowers and counterparties may default on their obligations to us.",
        "A deterioration in credit quality could increase our allowance for credit losses.",
        "Declines in collateral values may increase losses on our loan portfolio.",
        "Higher delinquencies and charge-offs in consumer loans would reduce earnings.",
        "Concentrations of credit exposure to a single counterparty increase loss severity.",
        "Commercial real estate loans may experience elevated default rates.",
    ],
    "market": [
        "Changes in interest rates could reduce our net interest income.",
        "Volatility in commodity prices affects the value of our production.",
        "Fluctuations in foreign currency exchange rates may lower reported revenue.",
        "Sustained inflation may increase our operating costs and reduce demand.",
        "Declines in equity markets could reduce the value of our investment securities.",
        "Lower crude oil and natural gas prices would reduce our cash flows.",
    ],
    "operational": [
        "Disruptions in our supply chain could delay product shipments.",
        "We depend on third-party vendors for critical manufacturing services.",
        "An outage at a key facility could interrupt our operations.",
        "Human error or failed internal processes may cause operational losses.",
        "The loss of key personnel could disrupt the execution of our strategy.",
        "Accidents at our facilities could cause injuries and business interruption.",
    ],
    "regulatory": [
        "We are subject to extensive regulation and supervision by government authorities.",
        "Changes in tax laws could increase our effective tax rate.",
        "Antitrust investigations could require changes to our business practices.",
        "Failure to comply with capital requirements could limit our ability to pay dividends.",
        "Litigation and regulatory enforcement actions may result in significant fines.",
        "New legislation could impose additional compliance costs on our operations.",
    ],
    "cyber": [
        "Cyberattacks could compromise the security of our information systems.",
        "A data breach could expose confidential customer information.",
        "Ransomware and malware may disrupt our networks and services.",
        "Unauthorized access to our systems could damage our reputation.",
        "Phishing campaigns target our employees to obtain credentials.",
        "Security vulnerabilities in third-party software could be exploited by attackers.",
    ],
    "climate": [
        "Climate change regulation could increase the cost of our operations.",
        "Policies to reduce greenhouse gas emissions may lower demand for our products.",
        "Extreme weather events could damage our facilities and disrupt production.",
        "The energy transition toward low-carbon sources may affect our long-term strategy.",
        "Carbon pricing mechanisms could increase our compliance costs.",
        "Investors increasingly scrutinize our net zero commitments and climate disclosures.",
    ],
}
FILLER = [
    "Any of these events could harm our financial condition.",
    "We cannot predict the timing or extent of these developments.",
    "Our mitigation efforts may not be sufficient.",
]
COMPANIES = [
    ("JPM", 19617, "JPMorgan Chase & Co.", "banking"),
    ("BAC", 70858, "Bank of America Corp", "banking"),
    ("XOM", 34088, "Exxon Mobil Corp", "energy"),
    ("CVX", 93410, "Chevron Corp", "energy"),
    ("MSFT", 789019, "Microsoft Corp", "technology"),
    ("AAPL", 320193, "Apple Inc", "technology"),
]


def make_passages(rng: random.Random) -> list[dict]:
    rows = []
    for ticker, cik, name, sector in COMPANIES:
        for i in range(12):
            cats = [CATEGORIES[i % 6]]
            if i >= 6:
                cats.append(rng.choice([c for c in CATEGORIES if c != cats[0]]))
            sents = []
            for c in cats:
                sents += rng.sample(TEMPLATES[c], 4)
            sents += rng.sample(FILLER, 2)
            rows.append(
                {
                    "passage_id": f"{ticker}-2024-{i:03d}",
                    "cik": cik,
                    "ticker": ticker,
                    "company": name,
                    "sector": sector,
                    "filing_date": "2024-02-15",
                    **{c: int(c in cats) for c in CATEGORIES},
                    "label_source": "synthetic",
                    "text": " ".join(sents),
                }
            )
    return rows


def make_html(rng: random.Random) -> str:
    paras = []
    for _ in range(40):
        cat = rng.choice(CATEGORIES)
        paras.append(" ".join(rng.sample(TEMPLATES[cat], 5) + rng.sample(FILLER, 2)))
    body = "\n".join(f'<p><span style="font-weight:normal">{p}</span></p>' for p in paras)
    return f"""<html><head><title>SYNTHETIC 10-K fixture</title></head><body>
<div style="display:none">hidden xbrl header</div>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>30</td></tr>
<tr><td>Item 2.</td><td>Properties</td><td>31</td></tr>
</table>
<p>Item 1. Business</p>
<p>This synthetic company makes fixtures for tests. It is not a real filer.</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>Risk Factors</p>
{body}
<p>12</p>
<p><b>Item 1B. Unresolved Staff Comments</b></p>
<p>None.</p>
<p><b>Item 2. Properties</b></p>
<p>We lease offices.</p>
</body></html>
"""


def main() -> None:
    rng = random.Random(42)
    rows = make_passages(rng)
    with (HERE / "passages.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "sample_10k.htm").write_text(make_html(rng))
    print(f"wrote {len(rows)} passages and sample_10k.htm")


if __name__ == "__main__":
    main()
