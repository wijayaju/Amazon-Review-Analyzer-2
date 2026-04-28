# Amazon Review Analyzer

A machine-learning project that classifies Amazon product reviews as **real** or **fake** based on their text content.

## Project Overview

This project reads Amazon reviews from CSV data, cleans the text, engineers features, trains multiple classifiers, and serves predictions through a Streamlit dashboard. The workflow covers data loading, feature engineering, model training/evaluation, and an interactive frontend for live inference.

## File Structure

```
Amazon-Review-Analyzer-2/
├── data/            # Raw and processed review data files
├── model/           # Trained joblib models, metadata, and BERT LoRA adapter files
├── src/             # Python scripts for data processing, training, and evaluation
├── webapp/          # Frontend application for serving model predictions
├── pyproject.toml   # Project metadata and dependencies (used by uv)
├── .gitignore       # Files and directories excluded from version control
└── README.md        # Project documentation
```

## Getting Started

### Prerequisites

This project uses **[uv](https://docs.astral.sh/uv/)** for Python project and environment management.

#### Install uv

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal so the `uv` command is available.

### Set Up the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/wijayaju/Amazon-Review-Analyzer-2.git
   cd Amazon-Review-Analyzer-2
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   uv sync
   ```

   This reads `pyproject.toml`, creates a `.venv` virtual environment, and installs all listed dependencies.

3. **Add new packages as needed**

   ```bash
   uv add <package-name>
   ```

4. **Run scripts**

   ```bash
   uv run python src/<script>.py
   ```

## Usage

1. Place the raw review data file in the `data/` directory.
2. Run preprocessing to create the engineered dataset:

   ```bash
   uv run python src/preprocess.py --input "data/<INPUT_CSV_PATH>.csv"
   ```

   Example:

   ```bash
   uv run python src/preprocess.py --input "data/fake-reviews.csv"
   ```

   Output is written to `data/preprocessed_reviews.csv`.
3. Use the scripts in `src/` to train the baseline TF-IDF + logistic regression model, the XGBoost model, or the BERT LoRA model.
4. Trained artifacts are saved in `model/` as `joblib` files, JSON metadata, and a `bert_lora/` adapter directory.
5. Launch the web application from `webapp/` to interact with the model through a browser.

### Run the Streamlit App

Start an interactive UI where you can paste a review and choose a model (`baseline`, `xgboost`, or `bert`) for instant classification:

```bash
uv run streamlit run webapp/streamlit_app.py
```

The app predicts whether the review is **AI-generated** or **human-written**, displays model confidence, and shows extracted feature values for the current input.

## Responsible AI Use

I used GitHub Copilot to help draft and scaffold parts of this project. I am the one responsible for reviewing, testing, and revising any AI-generated output before treating it as a final product. AI assistance is primarily used to accelerate development, but is not used to replace my own judgment.
