"""Fine-tune BERT with LoRA for fake review detection.

Assumptions:
- ``data/preprocessed_reviews.csv`` already exists.
- ``text_`` contains review text and ``label`` contains class labels.
- ``CG`` means fake and ``OR`` means real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "preprocessed_reviews.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "model" / "bert_lora"

LABEL_MAP = {"CG": 0, "OR": 1}
ID2LABEL = {0: "CG", 1: "OR"}


def load_data(input_path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(input_path)
    if "text_" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Input CSV must include 'text_' and 'label' columns")

    labels = frame["label"].map(LABEL_MAP)
    valid_rows = labels.notna() & frame["text_"].notna()
    texts = frame.loc[valid_rows, "text_"].astype(str)
    y = labels.loc[valid_rows].astype(int)
    return texts, y


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    positive_scores = probs[:, 1]
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    auc_score = roc_auc_score(labels, positive_scores)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score),
    }


def tokenize_batch(batch: dict[str, list[str]], tokenizer: AutoTokenizer, max_length: int) -> dict[str, list[int]]:
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def train_bert_lora(
    input_path: Path,
    model_dir: Path,
    model_name: str,
    test_size: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts, y = load_data(input_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    train_df = pd.DataFrame({"text": x_train.tolist(), "label": y_train.tolist()})
    test_df = pd.DataFrame({"text": x_test.tolist(), "label": y_test.tolist()})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    train_ds = train_ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length), batched=True)
    test_ds = test_ds.map(lambda batch: tokenize_batch(batch, tokenizer, max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)

    training_kwargs = {
        "output_dir": str(model_dir / "checkpoints"),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
    }

    # Transformers renamed this argument in newer versions.
    try:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=test_ds)

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    metadata = {
        "model_type": "bert_lora",
        "base_model": model_name,
        "pytorch_device": device,
        "label_mapping": LABEL_MAP,
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["query", "value"],
        },
    }

    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")

    return metadata["eval_metrics"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT with LoRA for fake review detection.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to preprocessed CSV.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory to save LoRA model.")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Hugging Face model name.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_metrics = train_bert_lora(
        input_path=args.input,
        model_dir=args.model_dir,
        model_name=args.model_name,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    print(f"Saved BERT LoRA model to {args.model_dir}")
    for key in ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_auc"]:
        if key in eval_metrics:
            print(f"{key}: {eval_metrics[key]:.4f}")


if __name__ == "__main__":
    main()