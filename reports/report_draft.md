# Predicting Steam Game Popularity from Storefront Metadata

## Abstract

This project studies whether Steam game popularity can be predicted from launch-visible storefront metadata. Using the March 2025 cleaned Steam Games Dataset, we define a binary task: predict whether a game falls in the top quartile of total review count among games released in the same year. We avoid leakage by excluding owner estimates, reviews, review scores, playtime, peak concurrent users, recommendations, Metacritic fields, and other post-release engagement variables from the input features. Our best model is a leakage-cleaned linear classifier using price/platform/release metadata plus store metadata such as genres, tags, categories, supported languages, developers, and publishers. On a temporal split that trains on 2009-2022 releases and tests on 2023-2025 releases, this model reaches PR-AUC 0.612 and ROC-AUC 0.804, substantially above the majority baseline PR-AUC of 0.251. Description text alone is weaker, and an SVD plus histogram gradient boosting model does not outperform the sparse linear metadata model.

## 1. Introduction

Steam contains a large and rapidly growing catalog of games, making discoverability and commercial success difficult to predict. Developers, publishers, and platform designers may want to estimate whether a game has traits associated with high post-release attention before large-scale launch investment. We frame this as a data mining problem: given launch-visible Steam store metadata, predict whether a game will be relatively popular within its release-year cohort.

The core challenge is target leakage. Steam datasets often include review counts, owner estimates, recommendations, playtime, and concurrent-user fields. These are strong measures of popularity, but they are not launch-time features. This project therefore uses post-release review count only to define the target, while excluding review-derived and engagement-derived columns from the feature matrix.

## 2. Related Work

Prior work motivates both the task and the modeling choices. De Luisa et al. study Steam popularity prediction from features such as price, supported languages, release timing, and genres. Vu and Bezemer examine game discoverability by comparing indie games to top-selling games, supporting the use of tags, descriptions, and similarity-based metadata. Ma studies broader success drivers in gaming and finds strong signal in Steam and community metrics, which helps motivate our leakage boundary: those post-release signals are useful but should not be used for launch-time prediction. XGBoost and LightGBM papers motivate gradient-boosted tree methods for sparse, high-dimensional prediction problems. A related Steam rating regression study also supports comparing linear models with tree-based models.

## 3. Dataset and EDA

We use `games_march2025_cleaned.csv` from the Steam Games Dataset 2025. The cleaned CSV contains 89,618 rows and 47 columns with no duplicate rows. Important fields include `name`, `release_date`, `price`, platform flags, descriptions, supported languages, developers, publishers, categories, genres, tags, review counts, owner estimates, playtime, and peak concurrent users.

The dataset is highly skewed. Review activity and owner estimates follow a long-tail distribution where a small number of games receive extremely high attention. Genre and tag frequencies are also imbalanced: common labels such as Indie, Casual, Action, Adventure, Simulation, Strategy, and RPG dominate. Missingness varies substantially across columns; for example, Metacritic URL and editorial reviews are mostly missing, while core store metadata is much more complete.

We initially considered `estimated_owners` as the target, but it is too coarse for this classification task. It has only 14 unique buckets and creates excessive ties, especially in recent years. We therefore use `num_reviews_total`, which has thousands of unique values and produces a stable within-year top-quartile class.

## 4. Predictive Task

For each game, we define:

```text
popular = 1 if num_reviews_total is at or above the 75th percentile among games released in the same year
popular = 0 otherwise
```

Games with invalid review-count sentinels are dropped. The final labeled dataset contains 53,162 games with a positive rate of 25.05%. We use a temporal split: training data contains games released from 2009 through 2022, and test data contains games released from 2023 through 2025. This split is stricter than a random split because it evaluates whether patterns learned from older Steam releases transfer to newer releases.

Primary metrics are PR-AUC and ROC-AUC. PR-AUC is especially important because only one quarter of games are labeled popular. We also report accuracy, precision, recall, and F1.

## 5. Models and Features

We remove leakage-prone fields including owner estimates, review counts, positive/negative review totals, recommendations, playtime, peak CCU, Metacritic fields, ratings, score ranks, achievements, screenshots, movies, raw app IDs, URLs, and raw release dates. The retained launch-visible feature groups are:

- Numeric metadata: required age, price, DLC count, platform flags, discount, release month.
- Store metadata text: genres, tags, categories, supported languages, developers, publishers.
- Description text: name, short description, detailed description, about-the-game text.

Text fields are concatenated and transformed with TF-IDF. Numeric fields are imputed and scaled. We compare a majority baseline, linear classifiers, feature-group ablations, and an SVD plus histogram gradient boosting model. The SVD model compresses sparse TF-IDF features before fitting a non-linear gradient boosting classifier.

## 6. Results

Baseline models on the temporal test split:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.717 | 0.458 | 0.702 | 0.554 | 0.563 | 0.781 |
| Linear SGD | 0.775 | 0.549 | 0.580 | 0.564 | 0.427 | 0.712 |
| Majority baseline | 0.749 | 0.000 | 0.000 | 0.000 | 0.251 | 0.500 |

Feature ablation and stronger-model results:

| Experiment | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Linear numeric + store metadata | 0.698 | 0.441 | 0.774 | 0.562 | 0.612 | 0.804 |
| Linear store metadata only | 0.691 | 0.435 | 0.774 | 0.557 | 0.609 | 0.797 |
| Linear all features | 0.715 | 0.458 | 0.746 | 0.568 | 0.596 | 0.799 |
| SVD + histogram gradient boosting, all features | 0.710 | 0.449 | 0.699 | 0.547 | 0.588 | 0.777 |
| Linear descriptions only | 0.638 | 0.380 | 0.708 | 0.495 | 0.489 | 0.723 |
| Linear numeric only | 0.728 | 0.461 | 0.510 | 0.484 | 0.472 | 0.689 |

The best model is the linear classifier using numeric plus store metadata features. Store metadata alone nearly matches it, suggesting that tags, genres, categories, languages, developers, and publishers carry most of the predictive signal. Adding description text slightly hurts PR-AUC, likely because descriptions are noisy and shift over time. The non-linear SVD gradient boosting model also underperforms the best sparse linear model, suggesting that the high-dimensional sparse metadata representation is better handled directly by a linear classifier than compressed into 100 dense components for this task.

## 7. Error Analysis

False positives include many games whose metadata resembles successful puzzle, hidden-object, or casual titles but whose actual review counts fall below the within-year popularity threshold. Examples include repeated `Archaeology` and `3D PUZZLE` titles. This suggests the model sometimes overweights genre/tag patterns without enough information about brand strength, production quality, or marketing reach.

False negatives include niche or viral titles whose metadata does not look broadly successful from launch-visible store fields alone. Examples include `Aero GPX`, `Hidden Cats in New York`, and several adult-themed titles. These cases suggest that community dynamics, external traffic, franchise/fandom effects, and platform featuring may matter but are not captured in the current feature set.

High-confidence correct positives include recognizable or franchise-backed games such as `Train Sim World 5`, `Dying Light`, `For Honor`, `Monster Hunter Wilds`, and `Mortal Kombat 1`, indicating that developer/publisher/tag metadata helps identify some obvious high-attention releases.

## 8. Discussion

The main takeaway is that launch-visible Steam metadata contains meaningful but limited signal for predicting future review activity. The best model roughly doubles PR-AUC over the majority baseline, but it still makes systematic mistakes on games where popularity is driven by signals absent from the dataset. Store metadata is more useful than long natural-language descriptions, and the temporal split shows that prediction is harder than a random split would suggest.

Our strongest limitation is that review count measures attention rather than revenue or player satisfaction. Another limitation is that release-year normalization removes age bias but still compares games under changing Steam market conditions. Finally, developer/publisher history is represented only as text, not as an explicit historical feature; adding leakage-safe prior success features may improve performance.

## 9. Conclusion

We built a leakage-aware Steam popularity prediction pipeline using the March 2025 Steam Games Dataset. A sparse linear model using launch-visible numeric and store metadata features achieved the best temporal test performance, with PR-AUC 0.612 and ROC-AUC 0.804. The results support the hypothesis that store metadata is predictive of relative Steam popularity, but also show that external demand, community effects, and quality signals remain important missing factors.

## Reproducibility

Run:

```bash
make eda
make baseline
make experiments
make errors
make demo
```

Main output files:

- `reports/tables/ablation_metrics.csv`
- `reports/tables/model_metrics.csv`
- `reports/tables/error_cases.csv`
- `reports/tables/experiment_metadata.json`
- `models/best_experiment_model.joblib`
