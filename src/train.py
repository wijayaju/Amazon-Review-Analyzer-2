"""Train and evaluate an XGBoost model for fake review detection.

Assumptions:
- ``data/preprocessed_reviews.csv`` already exists.
- ``label`` is the target column.
- ``CG`` means fake and ``OR`` means real.
- The preprocessing step already created the numeric feature columns used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "preprocessed_reviews.csv"
DEFAULT_MODEL_OUTPUT = ROOT_DIR / "model" / "xgboost_review_model.joblib"

FEATURE_COLUMNS = [
    "char_length",
    "word_count",
    "punctuation_ct",
    "is_extreme_star",
    "VERB",
    "ADV",
    "NOUN",
    "digit_ct",
    "uppercase_ct",
    "exclamation_ct",
    "question_ct",
    "unique_word_ct",
    "unique_word_ratio",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
]


def load_data(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(input_path)

    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    label_map = {"CG": 0, "OR": 1}
    labels = frame["label"].map(label_map)
    valid_rows = labels.notna()

    features = frame.loc[valid_rows, FEATURE_COLUMNS].copy()
    target = labels.loc[valid_rows].astype(int)

    return features, target


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )


def train_model(input_path: Path, model_output_path: Path, test_size: float) -> dict[str, float]:
    features, target = load_data(input_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    prediction_scores = model.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, prediction_scores)

    report = classification_report(
        y_test,
        predictions,
        target_names=["CG (fake)", "OR (real)"],
        digits=4,
    )
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc": test_auc,
    }

    feature_names_path = model_output_path.with_name(f"{model_output_path.stem}_feature_names.json")
    metadata_path = model_output_path.with_name(f"{model_output_path.stem}_metadata.json")

    metadata = {
        "test_auc_score": float(test_auc),
        "num_features": len(FEATURE_COLUMNS),
        "label_mapping": {"CG": 0, "OR": 1},
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_output_path,
    )

    feature_names_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved feature names to {feature_names_path}")
    print(f"Saved metadata to {metadata_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost model on preprocessed reviews.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the preprocessed reviews CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.input, args.model_out, args.test_size)

    print(f"Saved model to {args.model_out}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()