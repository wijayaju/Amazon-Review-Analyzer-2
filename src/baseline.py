"""Baseline text classifier for fake review detection.

Assumptions:
- ``data/preprocessed_reviews.csv`` already exists.
- ``text_`` contains review text and ``label`` contains class labels.
- ``CG`` means fake and ``OR`` means real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "preprocessed_reviews.csv"
DEFAULT_MODEL_OUT = ROOT_DIR / "model" / "baseline_tfidf_logreg.joblib"

LABEL_MAP = {"CG": 0, "OR": 1}


def load_data(input_path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(input_path)
    if "text_" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Input CSV must include 'text_' and 'label' columns")

    labels = frame["label"].map(LABEL_MAP)
    valid_rows = labels.notna() & frame["text_"].notna()
    texts = frame.loc[valid_rows, "text_"].astype(str)
    y = labels.loc[valid_rows].astype(int)
    return texts, y


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_baseline(input_path: Path, model_out: Path, test_size: float) -> dict[str, float]:
    texts, y = load_data(input_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, scores)

    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=["CG (fake)", "OR (real)"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_mapping": LABEL_MAP}, model_out)

    metadata = {
        "model_type": "baseline_tfidf_logreg",
        "test_auc_score": float(auc_score),
        "label_mapping": LABEL_MAP,
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }
    metadata_path = model_out.with_name(f"{model_out.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")

    return {"auc": auc_score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + logistic regression baseline model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to preprocessed CSV.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT, help="Path to save model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_baseline(args.input, args.model_out, args.test_size)
    print(f"Saved model to {args.model_out}")
    print(f"AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()