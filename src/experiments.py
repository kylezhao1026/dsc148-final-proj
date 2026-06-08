from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_classifier
from src.features import (
    add_engineered_columns,
    build_preprocessor,
    infer_feature_spec,
    select_feature_groups,
)
from src.load_data import DEFAULT_DATA_PATH, load_games, normalize_columns
from src.target import add_popularity_target
from src.train import temporal_split


ABLATIONS: dict[str, set[str]] = {
    "numeric_only": {"numeric"},
    "descriptions_only": {"descriptions"},
    "store_metadata_only": {"store_metadata"},
    "numeric_plus_store_metadata": {"numeric", "store_metadata"},
    "all_features": {"numeric", "descriptions", "store_metadata"},
}


def build_linear_estimator(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        max_iter=400,
        class_weight="balanced",
        random_state=random_state,
    )


def build_stronger_estimator(random_state: int) -> Pipeline:
    return Pipeline(
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
    )


def prepare_data(args: argparse.Namespace):
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
    train_idx, test_idx = temporal_split(df, test_fraction=args.test_fraction)
    return df, spec, target_info, train_idx, test_idx


def fit_and_score(name: str, spec, estimator, X_train, y_train, X_test, y_test) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        steps=[
            ("features", build_preprocessor(spec)),
            ("model", clone(estimator)),
        ]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_classifier(pipeline, X_test, y_test)
    metrics["experiment"] = name
    metrics["feature_count_numeric"] = len(spec.numeric_columns)
    metrics["feature_count_categorical"] = len(spec.categorical_columns)
    metrics["feature_count_text"] = len(spec.text_columns)
    return pipeline, metrics


def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    df, full_spec, target_info, train_idx, test_idx = prepare_data(args)
    X_train = df.loc[train_idx]
    y_train = df.loc[train_idx, "popular"]
    X_test = df.loc[test_idx]
    y_test = df.loc[test_idx, "popular"]

    rows: list[dict] = []
    best_pipeline = None
    best_name = ""
    best_pr_auc = -1.0

    for name, groups in ABLATIONS.items():
        spec = select_feature_groups(full_spec, groups)
        pipeline, metrics = fit_and_score(
            f"linear_{name}",
            spec,
            build_linear_estimator(args.random_state),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        rows.append(metrics)
        if metrics["pr_auc"] > best_pr_auc:
            best_pipeline = pipeline
            best_name = metrics["experiment"]
            best_pr_auc = metrics["pr_auc"]

    strong_pipeline, strong_metrics = fit_and_score(
        "svd_hist_gradient_boosting_all_features",
        full_spec,
        build_stronger_estimator(args.random_state),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    rows.append(strong_metrics)
    if strong_metrics["pr_auc"] > best_pr_auc:
        best_pipeline = strong_pipeline
        best_name = strong_metrics["experiment"]
        best_pr_auc = strong_metrics["pr_auc"]

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows).set_index("experiment").sort_values("pr_auc", ascending=False)
    results.to_csv(output_dir / "ablation_metrics.csv")

    if best_pipeline is not None:
        joblib.dump(best_pipeline, model_dir / "best_experiment_model.joblib")

    metadata = {
        "target": target_info.__dict__,
        "full_feature_spec": full_spec.__dict__,
        "best_experiment": best_name,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_years": sorted(int(y) for y in df.loc[train_idx, "_release_year"].unique()),
        "test_years": sorted(int(y) for y in df.loc[test_idx, "_release_year"].unique()),
        "ablations": {name: sorted(groups) for name, groups in ABLATIONS.items()},
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Steam popularity ablation experiments.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--target-column", default="num_reviews_total")
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-year-rows", type=int, default=50)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=148)
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    table = run_experiments(parse_args())
    print(table.round(4))
