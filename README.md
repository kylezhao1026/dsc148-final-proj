# DSC 148 Final Project

Steam game popularity prediction project.

## Data

Download the Kaggle CSV and place it here:

```text
data/games_march2025_cleaned.csv
```

Dataset page:

https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline works with pandas, scikit-learn, matplotlib, joblib, and streamlit. LightGBM and XGBoost are optional stronger models.

## Run EDA

```bash
python -m src.eda --data data/games_march2025_cleaned.csv
```

Outputs:

- `reports/tables/dataset_summary.csv`
- `reports/tables/missingness.csv`
- `reports/tables/target_by_year.csv`
- `reports/figures/*.png`

## Train Models

```bash
python -m src.train --data data/games_march2025_cleaned.csv
```

For a faster first pass:

```bash
python -m src.train --data data/games_march2025_cleaned.csv --models majority logistic_regression linear_svm_sgd
```

## Run Ablations and Error Analysis

```bash
python -m src.experiments --data data/games_march2025_cleaned.csv --target-column num_reviews_total
python -m src.error_analysis --data data/games_march2025_cleaned.csv --model models/best_experiment_model.joblib --target-column num_reviews_total
```

Outputs:

- `reports/tables/ablation_metrics.csv`
- `reports/tables/experiment_metadata.json`
- `reports/tables/error_cases.csv`
- `models/best_experiment_model.joblib`

Outputs:

- `reports/tables/model_metrics.csv`
- `reports/tables/classification_report.txt`
- `reports/tables/run_metadata.json`
- `models/best_model.joblib`

The default target preference is `num_reviews_total`, which gives a cleaner
within-year top-quartile label than coarse owner buckets. You can pass another
target explicitly:

```bash
python -m src.train --data data/games_march2025_cleaned.csv --target-column estimated_owners
```

## Demo

After training:

```bash
streamlit run app/streamlit_app.py
```
