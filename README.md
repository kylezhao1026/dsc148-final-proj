# Steam Game Popularity Predictor

DSC 148 final project predicting whether a Steam game will become popular from storefront metadata.

## Demo

Try the deployed Streamlit app:

https://dsc148-final-proj.streamlit.app/

The app lets users enter a hypothetical Steam game profile and returns the model's predicted probability that the game will rank in the top quartile of popularity for its release-year cohort.

## Project Summary

We use the Kaggle Steam Games Dataset 2025 to predict game popularity from launch-visible metadata such as price, release timing, platforms, genres, tags, categories, supported languages, and store description text.

The predictive task is binary classification:

- `popular = 1` if a game is in the top 25% of the selected popularity signal within its release year
- default popularity signal: `num_reviews_total`
- evaluation metrics: accuracy, precision, recall, F1, PR-AUC, and ROC-AUC

The feature pipeline avoids target leakage by dropping direct post-release popularity signals such as owners, reviews, recommendations, CCU, playtime, ratings, and scores.

## Dataset

The data file is too large to commit. To run the project locally, download the cleaned CSV from Kaggle and place it here:

```text
data/games_march2025_cleaned.csv
```

Dataset: https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

## Local Reproduction

The hosted demo is enough to try the model interactively. These commands are only needed if you want to reproduce the pipeline locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
make eda
make baseline
make experiments
make errors
```

To run the Streamlit app locally after training:

```bash
make demo
```

## Repository Guide

```text
src/                    model training, features, evaluation, EDA, error analysis
app/streamlit_app.py     Streamlit demo
reports/paper.pdf        final report
reports/paper.tex        report source
data/README.md           dataset placement note
Makefile                 common commands
```

## Report

The final report is available at `reports/paper.pdf`.
