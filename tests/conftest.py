"""Offline test harness: synthetic fixtures, hashing embeddings, a temp Chroma index.

Nothing here downloads a model or calls EDGAR, so it runs identically in CI.
"""

from __future__ import annotations

import os

os.environ.setdefault("EMBED_BACKEND", "hashing")
os.environ["OPENAI_API_KEY"] = ""  # always exercise the deterministic planner in tests

from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "passages.csv")


@pytest.fixture(scope="session")
def fixture_split(fixture_df, tmp_path_factory):
    from src.nlp.preprocess import load_split

    return load_split(fixture_df, path=tmp_path_factory.mktemp("split") / "split.json")


@pytest.fixture(scope="session")
def baseline_classifier(fixture_df, tmp_path_factory):
    from src.models.predict import BaselineClassifier
    from src.nlp.baseline import make_models
    from src.nlp.preprocess import label_matrix

    pipe = make_models()["logreg_ovr"]
    pipe.fit(fixture_df.text.tolist(), label_matrix(fixture_df))
    path = tmp_path_factory.mktemp("models") / "logreg.joblib"
    joblib.dump(pipe, path)
    return BaselineClassifier(path)


@pytest.fixture(scope="session")
def index_dir(fixture_df, baseline_classifier, tmp_path_factory) -> Path:
    from src.rag.index import build_index

    d = tmp_path_factory.mktemp("chroma")
    build_index(fixture_df, baseline_classifier, persist_dir=d, backend="hashing")
    return d


@pytest.fixture(scope="session")
def retriever(index_dir):
    from src.rag.retrieve import Retriever

    return Retriever(persist_dir=index_dir, backend="hashing")


@pytest.fixture(scope="session")
def wired_tools(retriever, baseline_classifier):
    from src.agent import tools

    tools.set_retriever(retriever)
    tools.set_classifier(baseline_classifier)
    yield tools
    tools.set_retriever(None)
    tools.set_classifier(None)
