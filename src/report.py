"""README figures, drawn only from results.json (never from hard-coded numbers).

python -m src.report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import CATEGORIES, NOTEBOOKS_DIR  # noqa: E402
from src.results import load  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow, magenta, green); fixed, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_class_f1() -> str | None:
    comp = load().get("comparison")
    if not comp:
        return None
    rows = comp["per_class_f1"]
    base_key = f"{comp['baseline_model']}_f1"
    base = [rows[c][base_key] for c in CATEGORIES]
    bert = [rows[c]["distilbert_f1"] for c in CATEGORIES]
    x = np.arange(len(CATEGORIES))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.bar(x - w / 2 - 0.01, base, w, color=SERIES[0], label=f"TF-IDF {comp['baseline_model']}")
    ax.bar(x + w / 2 + 0.01, bert, w, color=SERIES[1], label="DistilBERT")
    ax.set_xticks(x, CATEGORIES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (test split)", color=MUTED)
    ax.set_title("Per-class F1: baseline vs transformer", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    path = NOTEBOOKS_DIR / "per_class_f1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def retrieval_curves() -> str | None:
    ret = load().get("retrieval", {}).get("metrics")
    if not ret:
        return None
    ks = [1, 3, 5, 8]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), facecolor=SURFACE, sharey=True)
    for ax, metric in zip(axes, ["hit_rate", "mrr"], strict=True):
        _style(ax)
        for color, (name, row) in zip(SERIES, ret.items(), strict=False):
            ys = [row[f"{metric}@{k}"] for k in ks]
            ax.plot(ks, ys, color=color, linewidth=2, marker="o", markersize=6, label=name)
            ax.annotate(
                name.replace("_", " "),
                (ks[-1], ys[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=TEXT,
                fontsize=8,
                va="center",
            )
        ax.set_xticks(ks, [f"@{k}" for k in ks])
        ax.set_ylim(0, 1.02)
        ax.set_title(metric.replace("_", " "), color=TEXT, loc="left")
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = NOTEBOOKS_DIR / "retrieval.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def main() -> None:
    for fn in (per_class_f1, retrieval_curves):
        out = fn()
        print(f"{fn.__name__}: {out or 'skipped (not yet measured)'}")


if __name__ == "__main__":
    main()
