"""Shared multi-label metrics so every model is scored by identical code on identical rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import CATEGORIES, NOTEBOOKS_DIR  # noqa: E402


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, categories: list[str] = CATEGORIES
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {
        cat: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(s[i]),
        }
        for i, cat in enumerate(categories)
    }
    return {
        "per_class": per_class,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        # Subset accuracy: all six labels must be right. Reported because accuracy alone
        # is misleading under imbalance — macro-F1 is the headline number.
        "subset_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "n_test": int(y_true.shape[0]),
    }


def format_table(metrics_by_model: dict[str, dict]) -> str:
    models = list(metrics_by_model)
    header = f"{'class':<12}" + "".join(f"{m[:22]:>24}" for m in models)
    lines = [header, "-" * len(header)]
    for cat in CATEGORIES:
        row = f"{cat:<12}"
        for m in models:
            pc = metrics_by_model[m]["per_class"][cat]
            row += f"{pc['precision']:>8.2f}{pc['recall']:>8.2f}{pc['f1']:>8.2f}"
        lines.append(row)
    lines.append("-" * len(header))
    for key in ["macro_f1", "micro_f1", "subset_accuracy", "hamming_loss"]:
        lines.append(f"{key:<12}" + "".join(f"{metrics_by_model[m][key]:>24.4f}" for m in models))
    lines.insert(1, f"{'':<12}" + "".join(f"{'P':>8}{'R':>8}{'F1':>8}" for _ in models))
    return "\n".join(lines)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path | None = None
) -> Path:
    """One 2x2 confusion matrix per category (multi-label has no single NxN matrix)."""
    cms = multilabel_confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, cat, cm in zip(axes.flat, CATEGORIES, cms, strict=True):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > cm.max() / 2 else "black",
                fontsize=12,
            )
        ax.set_title(cat)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
    fig.suptitle(title)
    fig.tight_layout()
    out_path = out_path or NOTEBOOKS_DIR / f"confusion_{title.lower().replace(' ', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def compare_per_class(baseline: dict, transformer: dict, baseline_name: str) -> dict:
    """Per-class F1 comparison, naming the winner honestly — including baseline wins."""
    out = {}
    for cat in CATEGORIES:
        b = baseline["per_class"][cat]["f1"]
        t = transformer["per_class"][cat]["f1"]
        winner = "tie" if abs(b - t) < 1e-9 else ("transformer" if t > b else baseline_name)
        out[cat] = {
            f"{baseline_name}_f1": b,
            "distilbert_f1": t,
            "delta": round(t - b, 4),
            "winner": winner,
            "support": transformer["per_class"][cat]["support"],
        }
    out["macro"] = {
        f"{baseline_name}_f1": baseline["macro_f1"],
        "distilbert_f1": transformer["macro_f1"],
        "delta": round(transformer["macro_f1"] - baseline["macro_f1"], 4),
        "winner": "transformer"
        if transformer["macro_f1"] > baseline["macro_f1"]
        else baseline_name,
        "support": transformer["n_test"],
    }
    return out
