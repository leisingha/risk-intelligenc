"""Unsupervised view of the corpus: TF-IDF → TruncatedSVD (LSA) → KMeans.

Answers "what themes exist before we impose our six categories?" Each cluster is
described by its top terms (centroid mapped back through the SVD), its sector mix,
and, where labels exist, the dominant labelled category.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import Normalizer  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORIES,
    LABELS_PATH,
    NOTEBOOKS_DIR,
    PASSAGES_PATH,
    ROOT,
    SEED,
)
from src.nlp.preprocess import build_vectorizer  # noqa: E402


def cluster(texts: list[str], k: int = 6, n_components: int = 100, min_df: int = 2):
    vec = build_vectorizer(min_df)
    x = vec.fit_transform(texts)
    n_components = min(n_components, x.shape[1] - 1, x.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    z = Normalizer(copy=False).fit_transform(svd.fit_transform(x))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    assign = km.fit_predict(z)
    terms = np.array(vec.get_feature_names_out())
    centroids_tfidf = svd.inverse_transform(km.cluster_centers_)
    top_terms = {c: terms[np.argsort(centroids_tfidf[c])[::-1][:10]].tolist() for c in range(k)}
    return {
        "assign": assign,
        "z": z,
        "top_terms": top_terms,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "silhouette": float(silhouette_score(z, assign, random_state=SEED)),
        "n_components": n_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()
    df = pd.read_parquet(PASSAGES_PATH)
    res = cluster(df.text.tolist(), k=args.k)
    df["cluster"] = res["assign"]

    labels = pd.read_csv(LABELS_PATH) if LABELS_PATH.exists() else None
    clusters = {}
    for c in range(args.k):
        members = df[df.cluster == c]
        info = {
            "size": len(members),
            "top_terms": ", ".join(res["top_terms"][c]),
            "sector_mix": ", ".join(f"{s}:{n}" for s, n in members.sector.value_counts().items()),
        }
        if labels is not None:
            lab = labels[labels.passage_id.isin(members.passage_id)]
            if len(lab):
                counts = lab[CATEGORIES].sum().sort_values(ascending=False)
                info["dominant_label"] = f"{counts.index[0]} ({int(counts.iloc[0])}/{len(lab)})"
        clusters[f"cluster_{c}"] = info
        print(
            f"[{c}] n={info['size']:<4} {info['top_terms']}\n     {info['sector_mix']}"
            + (f" | {info['dominant_label']}" if "dominant_label" in info else "")
        )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(res["z"][:, 0], res["z"][:, 1], c=res["assign"], cmap="tab10", s=8)
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title(f"Risk passages: KMeans k={args.k} on LSA space")
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=8)
    fig.tight_layout()
    fig_path = NOTEBOOKS_DIR / "clusters.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    from src.results import record

    record(
        "clustering",
        {
            "k": args.k,
            "svd_components": res["n_components"],
            "svd_explained_variance": round(res["explained_variance"], 4),
            "silhouette": round(res["silhouette"], 4),
            "clusters": clusters,
            "figure": str(fig_path.relative_to(ROOT)),
        },
    )


if __name__ == "__main__":
    main()
