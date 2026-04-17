"""Simple preprocessing and feature engineering for the Amazon review dataset.

Assumptions:
- The raw CSV has a text column named ``text_``.
- Rows with empty review text are skipped.
- The raw CSV may include a numeric ``rating`` column.
- POS columns (``VERB``, ``ADV``, ``NOUN``) use lightweight suffix/rule heuristics.
"""

from __future__ import annotations

import argparse
import csv
import re
import string
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "fake-reviews.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "preprocessed_reviews.csv"

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

COMMON_VERBS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "done",
    "get",
    "gets",
    "got",
    "had",
    "has",
    "have",
    "is",
    "love",
    "loved",
    "make",
    "made",
    "need",
    "recommend",
    "seem",
    "was",
    "were",
    "will",
    "would",
}


def clean_text(text: str) -> str:
    """Lowercase text and collapse repeated whitespace."""

    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_verb(word: str) -> bool:
    return word in COMMON_VERBS or word.endswith(("ed", "ing"))


def _is_adv(word: str) -> bool:
    return word.endswith("ly")


def _is_noun(word: str) -> bool:
    return word.endswith(("tion", "ment", "ness", "ity", "ship", "er", "or"))


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def extract_features(raw_text: str, cleaned_text: str, rating_value: str | None) -> dict[str, int | float]:
    """Create simple numeric features for downstream modeling."""

    words = WORD_RE.findall(cleaned_text)
    word_count = len(words)
    unique_words = len(set(words))

    punctuation_ct = sum(1 for char in raw_text if char in string.punctuation)
    uppercase_ct = sum(1 for char in raw_text if char.isupper())
    digit_ct = sum(1 for char in raw_text if char.isdigit())
    exclamation_ct = raw_text.count("!")
    question_ct = raw_text.count("?")

    char_length = len(raw_text)
    avg_word_length = (sum(len(word) for word in words) / word_count) if word_count else 0.0
    unique_word_ratio = (unique_words / word_count) if word_count else 0.0

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
    sentence_count = len(sentences)
    avg_sentence_length = (word_count / sentence_count) if sentence_count else 0.0

    verb_ct = sum(1 for word in words if _is_verb(word))
    adv_ct = sum(1 for word in words if _is_adv(word))
    noun_ct = sum(1 for word in words if _is_noun(word))

    rating = _safe_float(rating_value)
    is_extreme_star = int(rating is not None and (rating <= 1.5 or rating >= 4.5))

    return {
        "char_length": char_length,
        "word_count": word_count,
        "punctuation_ct": punctuation_ct,
        "is_extreme_star": is_extreme_star,
        "VERB": verb_ct,
        "ADV": adv_ct,
        "NOUN": noun_ct,
        "digit_ct": digit_ct,
        "uppercase_ct": uppercase_ct,
        "exclamation_ct": exclamation_ct,
        "question_ct": question_ct,
        "unique_word_ct": unique_words,
        "unique_word_ratio": round(unique_word_ratio, 6),
        "avg_word_length": round(avg_word_length, 6),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 6),
    }


def preprocess_reviews(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> int:
    """Read the raw CSV, clean the review text, and write a new CSV.

    Returns the number of rows written.
    """

    with input_path.open(newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = list(reader.fieldnames or [])

        feature_columns = [
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

        if "clean_text" not in fieldnames:
            fieldnames.append("clean_text")

        for column in feature_columns:
            if column not in fieldnames:
                fieldnames.append(column)

        rows_written = 0

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                raw_text = (row.get("text_") or "").strip()
                if not raw_text:
                    continue

                cleaned_text = clean_text(raw_text)
                row["clean_text"] = cleaned_text
                row.update(extract_features(raw_text, cleaned_text, row.get("rating")))
                writer.writerow(row)
                rows_written += 1

    return rows_written


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing."""

    parser = argparse.ArgumentParser(description="Preprocess Amazon reviews into a feature CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the raw reviews CSV (default: data/fake-reviews.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = preprocess_reviews(input_path=args.input)
    print(f"Wrote {rows_written} cleaned rows to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()