from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_PATH = Path("data/games_march2025_cleaned.csv")


def load_games(path: str | Path = DEFAULT_DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Load the Steam games CSV with conservative memory settings."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download games_march2025_cleaned.csv "
            "from Kaggle and place it under data/."
        )
    return pd.read_csv(csv_path, low_memory=False, nrows=nrows)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case-ish column names."""
    out = df.copy()
    out.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        for col in out.columns
    ]
    return out


def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Find the first candidate that appears as an exact or substring match."""
    column_list = list(columns)
    candidate_list = [c.lower() for c in candidates]

    for candidate in candidate_list:
        if candidate in column_list:
            return candidate

    for candidate in candidate_list:
        for col in column_list:
            if candidate in col:
                return col

    return None


def parse_release_year(df: pd.DataFrame) -> pd.Series:
    """Infer release year from common Steam release-date columns."""
    date_col = find_first_column(
        df.columns,
        ["release_date", "released", "date_release", "release", "release_year"],
    )
    if date_col is None:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")

    if "year" in date_col:
        return pd.to_numeric(df[date_col], errors="coerce").astype("Int64")

    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return dates.dt.year.astype("Int64")


def parse_release_month(df: pd.DataFrame) -> pd.Series:
    date_col = find_first_column(df.columns, ["release_date", "released", "date_release"])
    if date_col is None:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return dates.dt.month.astype("Int64")
