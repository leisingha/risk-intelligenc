"""Classical multi-label baselines on TF-IDF features.

Three models, one vectorizer, one fixed split:
- Logistic Regression, one-vs-rest (linear, calibrated-ish probabilities, fast)
- Decision Tree (native multi-output; interpretable but high variance)
- Gradient Boosting, one-vs-rest (non-linear; slow on sparse high-dim features)
"""

from __future__ import annotations

import argparse
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import BASELINE_MODEL_PATH, CATEGORIES, ROOT, SEED
from src.models.evaluate import compute_metrics, format_table, plot_confusion
from src.nlp.preprocess import build_vectorizer, label_matrix, load_labelled, load_split


def make_models(min_df: int = 2) -> dict[str, Pipeline]:
    return {
        "logreg_ovr": Pipeline(
            [
                ("tfidf", build_vectorizer(min_df)),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED
                        )
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("tfidf", build_vectorizer(min_df)),
                (
                    "clf",
                    DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=SEED),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("tfidf", build_vectorizer(min_df)),
                (
                    "clf",
                    OneVsRestClassifier(
                        GradientBoostingClassifier(
                            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=SEED
                        )
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(
    train: pd.DataFrame, test: pd.DataFrame, min_df: int = 2, plot: bool = True
) -> dict[str, dict]:
    y_train, y_test = label_matrix(train), label_matrix(test)
    absent = [c for c, n in zip(CATEGORIES, y_train.sum(axis=0), strict=True) if n == 0]
    if absent:
        raise ValueError(
            f"Categories with no positive training examples: {absent}. "
            "Label more passages before training."
        )
    out: dict[str, dict] = {}
    for name, pipe in make_models(min_df).items():
        t0 = time.perf_counter()
        pipe.fit(train.text.tolist(), y_train)
        train_s = time.perf_counter() - t0
        y_pred = np.asarray(pipe.predict(test.text.tolist()), dtype=int)
        metrics = compute_metrics(y_test, y_pred)
        metrics["train_seconds"] = round(train_s, 2)
        metrics["n_features"] = len(pipe.named_steps["tfidf"].vocabulary_)
        if plot:
            metrics["confusion_figure"] = str(
                plot_confusion(y_test, y_pred, f"baseline {name}").relative_to(ROOT)
            )
        out[name] = {"pipeline": pipe, "metrics": metrics, "y_pred": y_pred}
    return out


def top_terms(pipe: Pipeline, n: int = 8) -> dict[str, list[str]]:
    """Highest-weighted features per class from the logistic regression: what TF-IDF learned."""
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    ovr = pipe.named_steps["clf"]
    terms = {}
    for cat, est in zip(CATEGORIES, ovr.estimators_, strict=True):
        if not hasattr(est, "coef_"):
            terms[cat] = []
            continue
        terms[cat] = vocab[np.argsort(est.coef_[0])[::-1][:n]].tolist()
    return terms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    df = load_labelled()
    train, test = load_split(df)
    print(f"Train {len(train)} / test {len(test)} passages (split seed {SEED})")
    results = train_and_evaluate(train, test)
    metrics = {name: r["metrics"] for name, r in results.items()}
    print(format_table(metrics))

    best = max(results, key=lambda n: results[n]["metrics"]["macro_f1"])
    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["logreg_ovr"]["pipeline"], BASELINE_MODEL_PATH)
    terms = top_terms(results["logreg_ovr"]["pipeline"])
    print(f"\nBest baseline by macro-F1: {best}")
    print(f"Saved logistic regression pipeline to {BASELINE_MODEL_PATH}")
    for cat, ts in terms.items():
        print(f"  {cat:<12} {', '.join(ts)}")

    if not args.no_record:
        from src.results import record

        record(
            "baseline",
            {
                "n_train": len(train),
                "n_test": len(test),
                "best_model": best,
                "macro_f1": {n: m["macro_f1"] for n, m in metrics.items()},
                **{f"{n}_per_class": m["per_class"] for n, m in metrics.items()},
                "summary": {
                    n: {
                        k: m[k]
                        for k in [
                            "macro_f1",
                            "micro_f1",
                            "subset_accuracy",
                            "hamming_loss",
                            "train_seconds",
                            "n_features",
                        ]
                    }
                    for n, m in metrics.items()
                },
                "confusion_figures": {n: m["confusion_figure"] for n, m in metrics.items()},
                "logreg_top_terms": {c: ", ".join(t) for c, t in terms.items()},
            },
        )


if __name__ == "__main__":
    main()
