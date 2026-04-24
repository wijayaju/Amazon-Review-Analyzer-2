"""Tune and evaluate an XGBoost model for fake review detection.

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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "preprocessed_reviews.csv"
DEFAULT_MODEL_OUTPUT = ROOT_DIR / "model" / "xgboost_tuned_model.joblib"
DEFAULT_PLOT_OUTPUT = ROOT_DIR / "model" / "xgboost_feature_importance.png"

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

    labels = frame["label"].map({"CG": 0, "OR": 1})
    valid_rows = labels.notna()

    features = frame.loc[valid_rows, FEATURE_COLUMNS].copy()
    target = labels.loc[valid_rows].astype(int)
    return features, target


def make_grid_search(cv_folds: int) -> GridSearchCV:
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "reg_alpha": [0.0, 0.5],
        "reg_lambda": [1.0, 2.0],
    }

    return GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )


def save_feature_importance_plot(model: XGBClassifier, output_path: Path) -> None:
    importance = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLUMNS, importance), key=lambda item: item[1], reverse=True)

    names = [name for name, _ in pairs]
    values = [value for _, value in pairs]

    plt.figure(figsize=(10, 6))
    plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def select_best_threshold(y_true: pd.Series, y_scores: pd.Series) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # precision/recall have one extra element compared to thresholds.
    best_threshold = 0.5
    best_f1 = -1.0
    for idx, threshold in enumerate(thresholds):
        p = precision[idx]
        r = recall[idx]
        denom = p + r
        f1_value = (2 * p * r / denom) if denom else 0.0
        if f1_value > best_f1:
            best_f1 = f1_value
            best_threshold = float(threshold)

    return best_threshold


def tune_model(
    input_path: Path,
    model_output_path: Path,
    plot_output_path: Path,
    test_size: float,
    cv_folds: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    features, target = load_data(input_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )

    grid_search = make_grid_search(cv_folds=cv_folds)
    grid_search.fit(x_train, y_train)
    best_model: XGBClassifier = grid_search.best_estimator_

    prediction_scores = best_model.predict_proba(x_test)[:, 1]
    best_threshold = select_best_threshold(y_test, prediction_scores)
    predictions = (prediction_scores >= best_threshold).astype(int)

    test_auc = roc_auc_score(y_test, prediction_scores)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc": test_auc,
    }

    selected_features = [
        feature
        for feature, importance in zip(FEATURE_COLUMNS, best_model.feature_importances_)
        if float(importance) > 0.0
    ]

    metadata = {
        "test_auc_score": float(test_auc),
        "num_features": len(FEATURE_COLUMNS),
        "selected_feature_count": len(selected_features),
        "label_mapping": {"CG": 0, "OR": 1},
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "best_threshold": float(best_threshold),
        "best_params": grid_search.best_params_,
    }

    metadata_path = model_output_path.with_name(f"{model_output_path.stem}_metadata.json")
    selected_features_path = model_output_path.with_name(f"{model_output_path.stem}_selected_features.json")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "selected_features": selected_features,
            "best_params": grid_search.best_params_,
            "best_threshold": best_threshold,
        },
        model_output_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    selected_features_path.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")
    save_feature_importance_plot(best_model, plot_output_path)

    print(f"Saved metadata to {metadata_path}")
    print(f"Saved selected features to {selected_features_path}")

    return metrics, grid_search.best_params_, best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune an XGBoost model using grid search.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to preprocessed CSV.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUTPUT, help="Path to save tuned model.")
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT_OUTPUT, help="Path to save importance plot.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds for grid search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, best_params, best_threshold = tune_model(
        input_path=args.input,
        model_output_path=args.model_out,
        plot_output_path=args.plot_out,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    print(f"Saved tuned model to {args.model_out}")
    print(f"Saved feature importance plot to {args.plot_out}")
    print(f"Best params: {best_params}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()