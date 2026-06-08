from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_classifier, write_classification_report, write_metrics_table
from src.features import add_engineered_columns, build_preprocessor, infer_feature_spec
from src.load_data import DEFAULT_DATA_PATH, load_games, normalize_columns
from src.target import add_popularity_target


def build_models(random_state: int = 148) -> dict[str, object]:
    models: dict[str, object] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            random_state=random_state,
        ),
        "linear_svm_sgd": SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "svd_hist_gradient_boosting": Pipeline(
            steps=[
                ("svd", TruncatedSVD(n_components=100, random_state=random_state)),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.06,
                        l2_regularization=0.05,
                        max_leaf_nodes=31,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return models


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.Index, pd.Index]:
    years = sorted(int(y) for y in df["_release_year"].dropna().unique())
    if len(years) < 3:
        return stratified_fallback_split(df, test_fraction)

    cutoff_position = max(1, int(round(len(years) * (1 - test_fraction))))
    cutoff_year = years[min(cutoff_position, len(years) - 1)]
    train_idx = df.index[df["_release_year"] < cutoff_year]
    test_idx = df.index[df["_release_year"] >= cutoff_year]

    if len(train_idx) < 100 or len(test_idx) < 50 or df.loc[test_idx, "popular"].nunique() < 2:
        return stratified_fallback_split(df, test_fraction)
    return train_idx, test_idx


def stratified_fallback_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.Index, pd.Index]:
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_fraction,
        stratify=df["popular"],
        random_state=148,
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def train_all(args: argparse.Namespace) -> pd.DataFrame:
    raw = load_games(args.data, nrows=args.nrows)
    df = normalize_columns(raw)
    df, target_info = add_popularity_target(
        df,
        target_column=args.target_column,
        quantile=args.quantile,
        min_year_rows=args.min_year_rows,
    )
    df = add_engineered_columns(df)

    spec = infer_feature_spec(df, raw_target_column=target_info.raw_target_column)
    preprocessor = build_preprocessor(spec)
    models = build_models(random_state=args.random_state)
    if args.models:
        requested = set(args.models)
        unknown = sorted(requested - set(models))
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(models)}")
        models = {name: model for name, model in models.items() if name in requested}

    train_idx, test_idx = temporal_split(df, test_fraction=args.test_fraction)
    X_train = df.loc[train_idx]
    y_train = df.loc[train_idx, "popular"]
    X_test = df.loc[test_idx]
    y_test = df.loc[test_idx, "popular"]

    metrics: dict[str, dict[str, float]] = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("features", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics[name] = evaluate_classifier(pipeline, X_test, y_test)

        score = metrics[name].get("pr_auc", -1.0)
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = write_metrics_table(metrics, output_dir / "model_metrics.csv")
    if best_pipeline is not None:
        joblib.dump(best_pipeline, model_dir / "best_model.joblib")
        write_classification_report(best_pipeline, X_test, y_test, output_dir / "classification_report.txt")

    metadata = {
        "target": target_info.__dict__,
        "feature_spec": spec.__dict__,
        "best_model": best_name,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_years": sorted(int(y) for y in df.loc[train_idx, "_release_year"].unique()),
        "test_years": sorted(int(y) for y in df.loc[test_idx, "_release_year"].unique()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Steam game popularity models.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to games_march2025_cleaned.csv")
    parser.add_argument("--target-column", default=None, help="Optional raw popularity target column.")
    parser.add_argument("--quantile", type=float, default=0.75, help="Within-year popularity cutoff.")
    parser.add_argument("--min-year-rows", type=int, default=50, help="Drop release years with too few rows.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Approximate test fraction.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick experiments.")
    parser.add_argument("--random-state", type=int, default=148)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset, e.g. --models majority logistic_regression linear_svm_sgd",
    )
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    table = train_all(parse_args())
    print(table.round(4))
