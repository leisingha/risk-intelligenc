"""Fine-tune distilbert-base-uncased for multi-label risk categorisation.

Multi-label, not multi-class: a passage about a ransomware attack that triggers an
SEC enforcement action is both `cyber` and `regulatory`. So:
  problem_type="multi_label_classification"  →  BCEWithLogitsLoss
  sigmoid per class at inference, threshold 0.5 — never softmax, which forces the six
  probabilities to sum to 1 and makes "two risks at once" impossible to express.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORIES, CLASSIFIER_BASE, DISTILBERT_DIR, SEED
from src.models.evaluate import compare_per_class, compute_metrics, format_table, plot_confusion
from src.nlp.preprocess import label_matrix, load_labelled, load_split

MAX_LENGTH = 256
THRESHOLD = 0.5


def _dataset(df: pd.DataFrame, tokenizer):
    from datasets import Dataset

    ds = Dataset.from_dict(
        {"text": df.text.tolist(), "labels": label_matrix(df).astype(np.float32).tolist()}
    )
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def train(train_df: pd.DataFrame, out_dir: Path = DISTILBERT_DIR) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(CATEGORIES)),
        label2id={c: i for i, c in enumerate(CATEGORIES)},
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=SEED,
        data_seed=SEED,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_dataset(train_df, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return {
        "train_seconds": round(seconds, 1),
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "max_length": MAX_LENGTH,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "model_size_mb": round(_dir_size_mb(out_dir), 1),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


class DistilBertClassifier:
    """Loads the fine-tuned model; returns sigmoid probabilities per category."""

    def __init__(self, model_dir: Path = DISTILBERT_DIR) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned model at {model_dir}; run " "`python -m src.models.finetune` first"
            )
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(**enc).logits
                out.append(self.torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(CATEGORIES)))


def evaluate(test_df: pd.DataFrame) -> tuple[dict, float]:
    clf = DistilBertClassifier()
    t0 = time.perf_counter()
    probs = clf.predict_proba(test_df.text.tolist())
    per_passage_ms = (time.perf_counter() - t0) / max(1, len(test_df)) * 1000
    y_pred = (probs >= THRESHOLD).astype(int)
    y_true = label_matrix(test_df)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion(y_true, y_pred, "distilbert")
    return metrics, per_passage_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate the saved model on the saved test split"
    )
    args = parser.parse_args()
    from src.results import load, record, update

    df = load_labelled()
    train_df, test_df = load_split(df, create=False)
    print(f"Train {len(train_df)} / test {len(test_df)} (same split as the baseline)")

    if not args.eval_only:
        info = train(train_df)
        print(
            f"Training took {info['train_seconds']}s; model {info['model_size_mb']} MB, "
            f"{info['parameters']:,} parameters"
        )
        record("transformer", info)

    previous = load().get("transformer", {}).get("metrics")
    metrics, ms = evaluate(test_df)
    if args.eval_only and previous is not None:
        current = {k: metrics[k] for k in previous}
        status = "MATCH" if current == previous else "DIFFERS"
        print(
            f"Reproducibility vs RESULTS.md: {status}\n  recorded {previous}\n  now      {current}"
        )
    update(
        "transformer",
        {
            "metrics": {
                k: metrics[k] for k in ["macro_f1", "micro_f1", "subset_accuracy", "hamming_loss"]
            },
            "per_class": metrics["per_class"],
            "inference_ms_per_passage_cpu": round(ms, 1),
            "threshold": THRESHOLD,
        },
    )

    baseline = load().get("baseline")
    if baseline is None:
        print(format_table({"distilbert": metrics}))
        return
    best = baseline["best_model"]
    base_metrics = {
        "per_class": baseline[f"{best}_per_class"],
        "macro_f1": baseline["macro_f1"][best],
        "micro_f1": baseline["summary"][best]["micro_f1"],
        "subset_accuracy": baseline["summary"][best]["subset_accuracy"],
        "hamming_loss": baseline["summary"][best]["hamming_loss"],
        "n_test": metrics["n_test"],
    }
    print(format_table({best: base_metrics, "distilbert": metrics}))
    comparison = compare_per_class(base_metrics, metrics, best)
    baseline_wins = [c for c in CATEGORIES if comparison[c]["winner"] == best]
    record(
        "comparison",
        {
            "baseline_model": best,
            "per_class_f1": comparison,
            "classes_where_baseline_wins": ", ".join(baseline_wins) or "none",
            "baseline_train_seconds": baseline["summary"][best]["train_seconds"],
            "distilbert_train_seconds": load()["transformer"].get("train_seconds", "n/a"),
        },
    )
    if baseline_wins:
        print(f"\nBaseline ({best}) beats DistilBERT on: {', '.join(baseline_wins)}")


if __name__ == "__main__":
    main()
