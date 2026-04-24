from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
import streamlit as st
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import clean_text, extract_features


MODEL_DIR = ROOT_DIR / "model"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_review_model.joblib"
BERT_MODEL_PATH = MODEL_DIR / "bert_lora"

LABEL_NAMES = {
    0: "AI-generated",
    1: "Human-written",
}


def _extract_model_from_artifact(artifact: Any) -> Any:
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


@st.cache_resource(show_spinner=False)
def load_baseline_model() -> Any:
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing baseline model file at {BASELINE_MODEL_PATH}")

    artifact = joblib.load(BASELINE_MODEL_PATH)
    return _extract_model_from_artifact(artifact)


@st.cache_resource(show_spinner=False)
def load_xgboost_model() -> tuple[Any, list[str]]:
    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost model file at {XGBOOST_MODEL_PATH}")

    artifact = joblib.load(XGBOOST_MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns")
    else:
        model = artifact
        feature_columns = None

    if model is None:
        raise ValueError("XGBoost artifact did not contain a model.")

    if not feature_columns:
        raise ValueError("XGBoost artifact did not contain feature_columns.")

    return model, list(feature_columns)


@st.cache_resource(show_spinner=False)
def load_bert_model() -> tuple[Any, Any]:
    if not BERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing BERT LoRA directory at {BERT_MODEL_PATH}")

    peft_config = PeftConfig.from_pretrained(str(BERT_MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_PATH))
    base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, str(BERT_MODEL_PATH))
    model.eval()
    return tokenizer, model


def classify_binary(predicted_label: int, human_prob: float) -> dict[str, Any]:
    ai_prob = 1.0 - human_prob
    winning_prob = human_prob if predicted_label == 1 else ai_prob
    confidence = max(0.0, min(1.0, winning_prob))

    return {
        "label": LABEL_NAMES[predicted_label],
        "confidence": confidence,
        "human_prob": max(0.0, min(1.0, human_prob)),
        "ai_prob": max(0.0, min(1.0, ai_prob)),
    }


def predict_with_baseline(review_text: str) -> dict[str, Any]:
    model = load_baseline_model()

    probs = model.predict_proba([review_text])[0]
    human_prob = float(probs[1])
    label = int(model.predict([review_text])[0])

    return classify_binary(predicted_label=label, human_prob=human_prob)


def predict_with_xgboost(review_text: str, rating: float) -> dict[str, Any]:
    model, feature_columns = load_xgboost_model()

    cleaned = clean_text(review_text)
    features = extract_features(review_text, cleaned, str(rating))
    feature_frame = pd.DataFrame([features])

    # Align one-row input with the exact feature order used by training.
    missing_columns = [column for column in feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required XGBoost features: {missing_columns}")

    feature_frame = feature_frame[feature_columns]

    probs = model.predict_proba(feature_frame)[0]
    human_prob = float(probs[1])
    label = int(model.predict(feature_frame)[0])

    return classify_binary(predicted_label=label, human_prob=human_prob)


def predict_with_bert(review_text: str) -> dict[str, Any]:
    import torch

    tokenizer, model = load_bert_model()

    encoded = tokenizer(
        review_text,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    human_prob = float(probs[1].item())
    label = int(torch.argmax(probs).item())

    return classify_binary(predicted_label=label, human_prob=human_prob)


def render_prediction(result: dict[str, Any], model_name: str) -> None:
    label = result["label"]
    confidence = result["confidence"]
    human_prob = result["human_prob"]
    ai_prob = result["ai_prob"]

    if label == "AI-generated":
        st.error(f"Prediction ({model_name}): {label}")
    else:
        st.success(f"Prediction ({model_name}): {label}")

    st.write(f"Confidence: {confidence:.2%}")
    st.progress(confidence)

    left_col, right_col = st.columns(2)
    left_col.metric("AI-generated probability", f"{ai_prob:.2%}")
    right_col.metric("Human-written probability", f"{human_prob:.2%}")


def render_feature_analysis(review_text: str, rating: float) -> None:
    cleaned = clean_text(review_text)
    features = extract_features(review_text, cleaned, str(rating))

    st.subheader("Feature Analysis")
    with st.expander("Show extracted features", expanded=True):
        st.write(f"Cleaned text: {cleaned}")
        feature_table = pd.DataFrame(
            {
                "feature": list(features.keys()),
                "value": list(features.values()),
            }
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Amazon Review Analyzer", page_icon="🛒", layout="centered")

    st.title("Amazon Review Analyzer")
    st.caption("Choose a model and check whether a review appears AI-generated or human-written.")

    model_choice = st.selectbox(
        "Model",
        ["baseline", "xgboost", "bert"],
        help="Select the trained model to run inference.",
    )

    rating = st.slider(
        "Optional star rating context (used by XGBoost features)",
        min_value=1.0,
        max_value=5.0,
        step=0.5,
        value=3.0,
    )

    review_text = st.text_area(
        "Paste a review",
        placeholder="Example: This product worked perfectly out of the box and feels premium.",
        height=200,
    )

    submitted = st.button("Classify Review", type="primary", use_container_width=True)

    if submitted:
        if not review_text.strip():
            st.warning("Please enter review text before running prediction.")
            return

        try:
            if model_choice == "baseline":
                result = predict_with_baseline(review_text)
            elif model_choice == "xgboost":
                result = predict_with_xgboost(review_text, rating)
            else:
                result = predict_with_bert(review_text)

            render_prediction(result, model_choice)
            render_feature_analysis(review_text, rating)
        except FileNotFoundError as error:
            st.error(str(error))
            st.info(
                "Model artifact not found. Train that model first, then rerun the app. "
                "Expected locations are model/baseline_tfidf_logreg.joblib, "
                "model/xgboost_review_model.joblib, and model/bert_lora/."
            )
        except Exception as error:  # noqa: BLE001
            st.exception(error)


if __name__ == "__main__":
    main()
