# Amazon Review Analyzer

A machine-learning project that classifies Amazon product reviews as **real** or **fake** based on their text content.

## Project Overview

This project reads a text file containing Amazon reviews, extracts features from the review text, trains a classification model, and serves predictions through a simple web application. The workflow covers data loading, feature engineering, model training/evaluation, and a lightweight frontend for interactive use.

## File Structure

```
Amazon-Review-Analyzer-2/
├── data/            # Raw and processed review data files
├── model/           # Trained model (.pkl) and feature-name files
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

1. Place your review data file in the `data/` directory.
2. Use the scripts in `src/` to preprocess data, train the model, and evaluate results.
3. Trained model artifacts will be saved to the `model/` directory.
4. Launch the web application from `webapp/` to interact with the model through a browser.

## Responsible AI Use

I used GitHub Copilot to help draft and scaffold parts of this project. I am the one responsible for reviewing, testing, and revising any AI-generated output before treating it as a final product. AI assistance is primarily used to accelerate development, but is not used to replace my own judgment.
