"""Text normalisation, tokenisation and TF-IDF features shared by every classical model.

Deliberately lemmatization-free: the stack has no lemmatizer (no NLTK/spaCy), and
unigram+bigram TF-IDF with sublinear tf is robust to inflection on this corpus.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import CATEGORIES, LABELS_PATH, PASSAGES_PATH, SEED, SPLIT_PATH

# Boilerplate that appears in nearly every risk factor and carries no signal.
DOMAIN_STOPWORDS = {
    "company",
    "companies",
    "business",
    "businesses",
    "could",
    "may",
    "might",
    "including",
    "result",
    "results",
    "adversely",
    "affect",
    "affected",
    "adverse",
    "material",
    "materially",
    "risk",
    "risks",
    "factors",
    "ability",
    "also",
    "us",
    "our",
    "we",
    "inc",
    "corp",
}
# scikit-learn's generic list contains words that carry risk signal here: "interest" would
# erase "interest rate", "system" would erase "information system", "fire", "bill", etc.
KEEP_WORDS = {"interest", "system", "systems", "fire", "bill", "amount", "full", "part", "call"}
STOPWORDS = frozenset((ENGLISH_STOP_WORDS - KEEP_WORDS) | DOMAIN_STOPWORDS)

_TOKEN_RE = re.compile(r"[a-z][a-z\-]+[a-z]")


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, drop numbers and punctuation, collapse whitespace."""
    text = text.lower().replace("’", "'").replace("—", " ").replace("–", " ")
    text = re.sub(r"\d+(\.\d+)?%?", " ", text)
    text = re.sub(r"[^a-z\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokens of 3+ characters, stopwords removed. Input should already be normalized."""
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


def build_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize,
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
    )


def load_labelled(labels_path=LABELS_PATH, passages_path=PASSAGES_PATH) -> pd.DataFrame:
    """Labels joined to passage text. labels.csv carries text too so it is self-contained."""
    labels = pd.read_csv(labels_path)
    missing = [c for c in ["passage_id", "text", *CATEGORIES] if c not in labels.columns]
    if missing:
        raise ValueError(f"{labels_path} is missing columns {missing}")
    labels[CATEGORIES] = labels[CATEGORIES].astype(int)
    return labels


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[CATEGORIES].to_numpy(dtype=int)


def _stratum(row: pd.Series, rarity: list[str]) -> str:
    """Multi-label rows can't be stratified directly; stratify on the rarest positive label."""
    for cat in rarity:
        if row[cat] == 1:
            return cat
    return "none"


def make_split(df: pd.DataFrame, test_size: float = 0.2, path=SPLIT_PATH) -> dict[str, list[str]]:
    rarity = sorted(CATEGORIES, key=lambda c: df[c].sum())
    strata = df.apply(_stratum, axis=1, rarity=rarity)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "rare")  # singletons can't be stratified
    train_ids, test_ids = train_test_split(
        df.passage_id.tolist(), test_size=test_size, random_state=SEED, stratify=strata
    )
    split = {
        "seed": SEED,
        "test_size": test_size,
        "train": sorted(train_ids),
        "test": sorted(test_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")
    return split


def load_split(
    df: pd.DataFrame, path=SPLIT_PATH, create: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The split is persisted so the baseline and the transformer see identical test rows."""
    if path.exists():
        split = json.loads(path.read_text())
        known = set(df.passage_id)
        if not set(split["train"]) | set(split["test"]) <= known:
            raise ValueError(
                f"{path} references passages not in the labelled set; delete it "
                "and re-run the baseline to regenerate the split"
            )
    elif create:
        split = make_split(df, path=path)
    else:
        raise FileNotFoundError(f"{path} not found; run `python -m src.nlp.baseline` first")
    train = df[df.passage_id.isin(split["train"])].reset_index(drop=True)
    test = df[df.passage_id.isin(split["test"])].reset_index(drop=True)
    return train, test


# Words that make a question a question rather than a topic ("what does X say about...").
QUERY_STOPWORDS = frozenset(
    {
        "say",
        "says",
        "said",
        "describe",
        "describes",
        "disclose",
        "discloses",
        "disclosed",
        "corpus",
        "mention",
        "mentions",
        "tell",
        "explain",
        "filing",
        "filings",
        "annual",
        "report",
        "reports",
        "regarding",
        "related",
        "relate",
        "concerning",
        "face",
        "faces",
        "compare",
        "comparison",
        "versus",
        "between",
        "differ",
        "difference",
    }
)


def fold_plural(token: str) -> str:
    """Plural folding (rates→rate, liabilities→liability) plus one domain compound
    (cybersecurity/cyberattacks→cyber). Used for term matching in reranking and grounding,
    not for TF-IDF features, which stay lemmatization-free."""
    if token.startswith("cyber"):
        return "cyber"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token


def content_terms(text: str, drop: frozenset[str] | set[str] = frozenset()) -> set[str]:
    """Plural-folded content words of a query or sentence, minus question filler."""
    return {fold_plural(t) for t in tokenize(normalize(text)) if t not in QUERY_STOPWORDS} - set(
        drop
    )
