"""Bootstrap multi-label risk categories with keyword rules, ready for hand correction.

Workflow (spec Phase 2.1):
  1. `python -m src.nlp.labels --bootstrap` samples 300 passages (stratified by company,
     seed 42), applies the keyword rules below and writes data/labels/labels.csv with
     label_source="keyword".
  2. A person reads each row, fixes the six 0/1 columns and sets label_source="hand".
  3. `python -m src.nlp.labels --stats` records category counts and co-occurrence.

A rule fires when a category's patterns match at least `min_hits` times, so a single
passing mention ("...including cyber attacks, pandemics and...") does not label a
passage that is really about something else.
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from src.config import CATEGORIES, LABELS_PATH, PASSAGES_PATH, SEED

KEYWORD_RULES: dict[str, tuple[list[str], int]] = {
    "credit": (
        [
            r"credit (risk|quality|loss|losses|exposure|rating)",
            r"\bdefault(s|ed)?\b",
            r"counterpart(y|ies)",
            r"borrower",
            r"\bloan(s)?\b",
            r"allowance for (credit|loan)",
            r"collateral",
            r"delinquen",
            r"charge-?offs?",
        ],
        2,
    ),
    "market": (
        [
            r"interest rate",
            r"market (risk|volatility|conditions|price)",
            r"foreign (currency|exchange)",
            r"commodity price",
            r"(oil|natural gas|crude) price",
            r"\bvolatil",
            r"\binflation",
            r"liquidity",
            r"equity (price|market)",
            r"exchange rate",
        ],
        2,
    ),
    "operational": (
        [
            r"operational (risk|failure|disruption)",
            r"supply chain",
            r"\boutage",
            r"human error",
            r"third[- ]party (vendor|provider|service)",
            r"\bvendor",
            r"business continuity",
            r"manufactur",
            r"\baccident",
            r"key (personnel|employees)",
            r"disruption",
        ],
        2,
    ),
    "regulatory": (
        [
            r"regulat",
            r"\blegislation",
            r"\bcompliance",
            r"\bsupervis",
            r"capital requirement",
            r"\bbasel\b",
            r"dodd-frank",
            r"antitrust",
            r"\blitigation",
            r"\bsanction",
            r"government(al)? (action|investigation|authorit)",
            r"\btax (law|polic|rate)",
        ],
        2,
    ),
    "cyber": (
        [
            r"cyber",
            r"information security",
            r"data breach",
            r"\bhack",
            r"ransomware",
            r"malware",
            r"unauthorized access",
            r"security (breach|incident|vulnerabilit)",
            r"phishing",
            r"denial[- ]of[- ]service",
        ],
        1,
    ),
    "climate": (
        [
            r"climate",
            r"greenhouse gas",
            r"\bemission",
            r"carbon",
            r"energy transition",
            r"low-carbon",
            r"net[- ]zero",
            r"extreme weather",
            r"\bparis agreement",
            r"renewable",
            r"sea level",
        ],
        1,
    ),
}

_COMPILED = {
    cat: ([re.compile(p, re.I) for p in pats], n) for cat, (pats, n) in KEYWORD_RULES.items()
}


def keyword_labels(text: str) -> dict[str, int]:
    out = {}
    for cat, (patterns, min_hits) in _COMPILED.items():
        hits = sum(len(p.findall(text)) for p in patterns)
        out[cat] = int(hits >= min_hits)
    return out


def bootstrap(n: int = 300, passages_path=PASSAGES_PATH, out_path=LABELS_PATH) -> pd.DataFrame:
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} exists and may contain hand corrections; "
            "move it aside explicitly before re-bootstrapping"
        )
    passages = pd.read_parquet(passages_path)
    if len(passages) < n:
        raise ValueError(f"Only {len(passages)} passages available, need {n}")
    # Stratify by company so no single filer dominates the labelled set.
    per_company = max(1, n // passages.ticker.nunique())
    sample = passages.groupby("ticker", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_company), random_state=SEED)
    )
    if len(sample) < n:
        rest = passages.drop(sample.index).sample(n - len(sample), random_state=SEED)
        sample = pd.concat([sample, rest])
    sample = sample.sample(frac=1, random_state=SEED).head(n)
    rows = []
    for _, r in sample.iterrows():
        rows.append(
            {
                "passage_id": r.passage_id,
                "ticker": r.ticker,
                "sector": r.sector,
                **keyword_labels(r.text),
                "label_source": "keyword",
                "text": r.text,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def stats(labels_path=LABELS_PATH, record_results: bool = True) -> dict:
    df = pd.read_csv(labels_path)
    y = df[CATEGORIES].astype(int)
    cooc = y.T @ y
    result = {
        "labelled_passages": len(df),
        "hand_corrected": int((df.label_source == "hand").sum()),
        "keyword_only": int((df.label_source == "keyword").sum()),
        "positives_per_category": {c: int(y[c].sum()) for c in CATEGORIES},
        "passages_with_no_label": int((y.sum(axis=1) == 0).sum()),
        "mean_labels_per_passage": round(float(y.sum(axis=1).mean()), 3),
        "co_occurrence": {c: {c2: int(cooc.loc[c, c2]) for c2 in CATEGORIES} for c in CATEGORIES},
    }
    if record_results:
        from src.results import record

        record("dataset", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("-n", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap:
        df = bootstrap(args.n)
        print(f"Wrote {len(df)} keyword-labelled rows to {LABELS_PATH}; hand-correct them next.")
    if args.stats or not args.bootstrap:
        s = stats()
        for k, v in s.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
