from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import predict_scores
from src.features import add_engineered_columns, infer_feature_spec
from src.load_data import DEFAULT_DATA_PATH, load_games, normalize_columns
from src.target import add_popularity_target
from src.train import temporal_split


DISPLAY_COLUMNS = [
    "appid",
    "name",
    "release_date",
    "price",
    "genres",
    "tags",
    "categories",
    "supported_languages",
    "num_reviews_total",
    "estimated_owners",
    "_popularity_value",
    "popular",
    "predicted_probability",
]


def build_error_cases(df: pd.DataFrame, scores, limit: int) -> pd.DataFrame:
    out = df.copy()
    out["predicted_probability"] = scores
    out["predicted_label"] = (out["predicted_probability"] >= 0.5).astype(int)
    out["error_type"] = "correct"
    out.loc[(out["popular"] == 0) & (out["predicted_label"] == 1), "error_type"] = "false_positive"
    out.loc[(out["popular"] == 1) & (out["predicted_label"] == 0), "error_type"] = "false_negative"

    false_positives = out[out["error_type"] == "false_positive"].sort_values(
        "predicted_probability", ascending=False
    ).head(limit)
    false_negatives = out[out["error_type"] == "false_negative"].sort_values(
        "predicted_probability", ascending=True
    ).head(limit)
    strong_correct = out[out["error_type"] == "correct"].assign(
        confidence=lambda x: x["predicted_probability"].where(
            x["predicted_label"].eq(1), 1 - x["predicted_probability"]
        )
    ).sort_values("confidence", ascending=False).head(limit)

    cases = pd.concat([false_positives, false_negatives, strong_correct], ignore_index=True)
    keep = [col for col in DISPLAY_COLUMNS + ["predicted_label", "error_type"] if col in cases.columns]
    return cases[keep]


def run_error_analysis(args: argparse.Namespace) -> pd.DataFrame:
    raw = load_games(args.data, nrows=args.nrows)
    df = normalize_columns(raw)
    df, target_info = add_popularity_target(
        df,
        target_column=args.target_column,
        quantile=args.quantile,
        min_year_rows=args.min_year_rows,
    )
    df = add_engineered_columns(df)
    infer_feature_spec(df, raw_target_column=target_info.raw_target_column)
    _, test_idx = temporal_split(df, test_fraction=args.test_fraction)
    test = df.loc[test_idx].copy()

    model = joblib.load(args.model)
    scores = predict_scores(model, test)
    cases = build_error_cases(test, scores, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_path, index=False)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model error-analysis cases.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", default="models/best_experiment_model.joblib")
    parser.add_argument("--target-column", default="num_reviews_total")
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-year-rows", type=int, default=50)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="reports/tables/error_cases.csv")
    return parser.parse_args()


if __name__ == "__main__":
    table = run_error_analysis(parse_args())
    print(table[["name", "popular", "predicted_probability", "predicted_label", "error_type"]].to_string())
