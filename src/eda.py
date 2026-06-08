from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.load_data import DEFAULT_DATA_PATH, load_games, normalize_columns
from src.target import add_popularity_target


def save_missingness(df: pd.DataFrame, output_dir: Path) -> None:
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(output_dir / "missingness.csv", index=False)


def save_basic_stats(df: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    pd.Series(summary).to_csv(output_dir / "dataset_summary.csv", header=["value"])

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.describe().T.to_csv(output_dir / "numeric_summary.csv")


def save_target_by_year(df: pd.DataFrame, output_dir: Path, figure_dir: Path) -> None:
    yearly = (
        df.groupby("_release_year")
        .agg(
            games=("popular", "size"),
            positive_rate=("popular", "mean"),
            median_popularity=("_popularity_value", "median"),
            p90_popularity=("_popularity_value", lambda s: s.quantile(0.9)),
        )
        .reset_index()
        .sort_values("_release_year")
    )
    yearly.to_csv(output_dir / "target_by_year.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(yearly["_release_year"], yearly["games"], color="#4c78a8", label="Games")
    ax1.set_xlabel("Release year")
    ax1.set_ylabel("Games")
    ax2 = ax1.twinx()
    ax2.plot(yearly["_release_year"], yearly["median_popularity"], color="#f58518", label="Median popularity")
    ax2.set_ylabel("Median raw popularity")
    fig.tight_layout()
    fig.savefig(figure_dir / "games_and_popularity_by_year.png", dpi=160)
    plt.close(fig)


def save_top_tokens(df: pd.DataFrame, output_dir: Path, figure_dir: Path) -> None:
    candidate_cols = [c for c in ["genres", "tags", "categories", "supported_languages"] if c in df.columns]
    rows = []
    for col in candidate_cols:
        tokens = (
            df[col]
            .dropna()
            .astype(str)
            .str.replace(r"[\[\]'{}\"]", "", regex=True)
            .str.split(r"[,;/|]", regex=True)
            .explode()
            .str.strip()
        )
        counts = tokens[tokens.ne("")].value_counts().head(25)
        for token, count in counts.items():
            rows.append({"column": col, "token": token, "count": int(count)})

        if not counts.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            counts.sort_values().plot(kind="barh", ax=ax, color="#54a24b")
            ax.set_title(f"Top {col}")
            ax.set_xlabel("Count")
            fig.tight_layout()
            fig.savefig(figure_dir / f"top_{col}.png", dpi=160)
            plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "top_metadata_tokens.csv", index=False)


def run_eda(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw = load_games(args.data, nrows=args.nrows)
    df = normalize_columns(raw)
    save_basic_stats(df, output_dir)
    save_missingness(df, output_dir)

    target_df, target_info = add_popularity_target(
        df,
        target_column=args.target_column,
        quantile=args.quantile,
        min_year_rows=args.min_year_rows,
    )
    save_target_by_year(target_df, output_dir, figure_dir)
    save_top_tokens(target_df, output_dir, figure_dir)
    (output_dir / "target_info.txt").write_text(str(target_info), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for Steam game popularity project.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-year-rows", type=int, default=50)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--figure-dir", default="reports/figures")
    return parser.parse_args()


if __name__ == "__main__":
    run_eda(parse_args())
