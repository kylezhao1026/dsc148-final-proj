# DSC 148 Final Project Plan

## Working title

Predicting Steam Game Popularity from Storefront Metadata

## Project thesis

Can we predict whether a Steam game will become popular using only game metadata that is plausibly available near launch, such as price, release timing, genres, tags, categories, supported languages, platforms, developer/publisher history, and store description text?

This is a good fit for the course requirements because the Kaggle Steam Games Dataset 2025 has 90,000+ games and 186 columns, enough for EDA, feature engineering, classification/regression baselines, stronger tree models, ablations, and a small interactive demo.

## Dataset

Primary dataset: Steam Games Dataset 2025, `games_march2025_cleaned.csv`.

Source: https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

Important dataset facts:

- Snapshot is up to date as of March 2025.
- Cleaned file removes duplicate and playtest versions.
- Kaggle page reports 90,000+ Steam games, 186 columns, and a 468 MB cleaned CSV.
- Data comes from Steam store scraping, Steam API, and Steam Spy.

EDA should include:

- Dataset size, missingness, duplicate checks, release-year coverage.
- Distribution of popularity proxies: estimated owners, review counts, positive ratio, peak concurrent users, average/median playtime if present.
- Price distribution, free-to-play share, discounts if present.
- Genre/tag/category frequencies and long-tail behavior.
- Release timing trends: number of games per year/month and saturation after Steam Direct.
- Popularity imbalance: top games likely dominate reviews/owners.
- Indie vs non-indie comparison if tag/category exists.
- Correlations among reviews, owners, price, supported languages, release age, and tags.

## Predictive task

Recommended task: binary classification.

Target: `popular = 1` if a game is in the top 25% of post-release popularity within its release-year cohort. After inspecting the real CSV, `num_reviews_total` is the best default target because it has thousands of unique values and produces a stable 25% positive class by year. Coarse owner buckets create too many ties in recent years.

Why release-year cohort normalization matters:

- Older games had more time to accumulate owners and reviews.
- A raw top-quartile threshold would over-reward older titles.
- Cohort ranking makes the task closer to predicting relative launch-market success.

Primary evaluation:

- ROC-AUC for ranking quality.
- F1 and PR-AUC because popularity will likely be imbalanced.
- Accuracy only as a secondary metric.

Validation split:

- Prefer temporal split: train on older releases, validate on recent releases, test on the newest held-out period.
- If release dates are too messy, use stratified train/validation/test split and report this limitation.

Leakage rule:

- Do not use features that directly define the target. If target is review count, exclude positive/negative reviews, review score, owners, CCU, and playtime from the feature set.
- Run an optional "post-release signals" experiment separately to show how much review/CCU features improve prediction, but label it as non-launch-time prediction.

## Baselines

Use at least three baselines:

1. Majority-class or popularity-prior baseline.
2. Logistic regression with numeric features and one-hot/multi-hot metadata.
3. Naive Bayes or linear SVM on TF-IDF description text.
4. Random forest as a non-boosted tree baseline.

These are defensible because they test simple class imbalance, linear metadata effects, text-only signal, and basic nonlinear interactions.

## Proposed model

Main model: LightGBM or XGBoost on tabular + sparse categorical/text features.

Feature groups:

- Numeric: price, required age, number of supported languages, number of screenshots/movies if present, release year/month, days since release for non-launch experiments.
- Multi-hot: genres, categories, tags, platforms.
- Text: TF-IDF or sentence embeddings from short/long descriptions, reduced with SVD if needed.
- Developer/publisher history: previous number of released games, previous median popularity, previous hit count, computed only from earlier release dates to avoid leakage.

Model comparison:

- Logistic regression on metadata only.
- Text-only TF-IDF model.
- Random forest.
- XGBoost.
- LightGBM.
- Final ensemble or calibrated LightGBM if it clearly wins.

Interpretability:

- Feature importance and/or SHAP.
- Ablation table by feature group: numeric only, tags only, text only, developer history only, all features.
- Case studies: high-confidence correct predictions and high-confidence errors.

## Literature framing

Use the provided readings like this:

- Vu and Bezemer, "Improving the Discoverability of Indie Games by Leveraging their Similarity to Top-Selling Games" motivates tag, genre, and description similarity as useful game-discoverability signals.
- Ma, "Unveiling Success Drivers in Gaming" finds Steam metrics such as followers, peak CCU, and total reviews are strong revenue predictors, with Twitch and Metacritic also useful. We cannot use all of those in a launch-time model, but the paper helps frame success drivers and leakage boundaries.
- De Luisa et al., "Predicting the Popularity of Games on Steam" directly motivates predicting Steam popularity from early game features such as price, size, languages, release timing, and genres.
- Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" supports using boosted trees for sparse tabular data.
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" supports LightGBM for large, sparse, high-dimensional features like multi-hot tags and TF-IDF.
- Teja et al., "Predicting Steam Games Rating with Regression" gives a related Steam prediction study where tree-based models performed well.

## Results to produce

Required result tables/figures:

- EDA: target distribution by year, genre/tag frequencies, popularity skew.
- Main performance table: all baselines plus proposed models.
- Ablation table: feature group contribution.
- Hyperparameter sensitivity: number of trees, learning rate, max depth/num leaves.
- Error analysis: examples of false positives and false negatives.
- Interpretation: top features and whether they match prior literature.

Good project claim if results cooperate:

"A gradient-boosted model using tags, release timing, price, supported languages, store text, and developer history predicts within-year Steam popularity better than linear and text-only baselines, but errors concentrate around viral indie hits and franchise-backed games whose marketing/community signals are not visible in basic storefront metadata."

## Demo idea

Build a small Streamlit demo:

- User enters a hypothetical Steam game: title, price, release month, tags, genres, categories, supported languages, and short description.
- Demo outputs predicted probability of being top-quartile popular.
- Show nearest successful games by tag/text similarity.
- Show top contributing features for the prediction.

This is feasible and aligns with the bonus-point working-demo suggestion.

## Division of work for two people

Person A:

- Download/load data.
- Clean columns and build target.
- EDA figures and dataset section.
- Implement baselines.

Person B:

- Literature review.
- Feature engineering for tags/text/developer history.
- Implement LightGBM/XGBoost and ablations.
- Demo and results writeup.

Joint work:

- Decide final target after inspecting actual columns.
- Write predictive task and model justification.
- Polish ACM-format report.
- Validate that no target leakage is present.

## Timeline

Day 1:

- Download dataset, inspect schema, pick target column.
- Create clean notebook/script and initial EDA.
- Confirm train/test split and leakage rules.

Day 2:

- Build preprocessing pipeline.
- Train majority/logistic/text/random-forest baselines.
- Draft dataset and predictive-task report sections.

Day 3:

- Train LightGBM/XGBoost.
- Run feature ablations and hyperparameter sweep.
- Generate result tables.

Day 4:

- Error analysis and SHAP/feature importance.
- Build Streamlit demo.
- Draft literature and model sections.

Day 5:

- Final report in ACM template.
- Re-run all notebooks/scripts from scratch.
- Clean GitHub repo with README, requirements, and instructions.

## Repo structure

```text
.
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_models_ablation.ipynb
├── src/
│   ├── load_data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   └── paper.tex
└── requirements.txt
```

## Immediate next steps

1. Download `games_march2025_cleaned.csv`.
2. Run a schema inspection and pick the cleanest target among estimated owners, total reviews, or peak CCU.
3. Create `data/README.md` documenting that the CSV is too large for GitHub and must be downloaded from Kaggle.
4. Start with the binary classification task; add regression only if the target looks clean after log transform.
