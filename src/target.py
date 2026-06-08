from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.load_data import find_first_column, parse_release_year


TARGET_CANDIDATES = [
    "num_reviews_total",
    "total_reviews",
    "review_count",
    "reviews_total",
    "estimated_owners",
    "owners",
    "recommendations",
    "peak_ccu",
    "ccu",
]


@dataclass(frozen=True)
class TargetInfo:
    target_column: str
    raw_target_column: str
    release_year_column: str
    threshold_quantile: float
    positive_rate: float
    rows: int


def parse_numeric_popularity(value: object) -> float:
    """Parse Steam popularity values, including owner ranges like '20,000 - 50,000'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if value >= 0 else np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    nums = [float(n.replace(",", "")) for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)]
    if not nums:
        return np.nan
    if len(nums) >= 2 and ("-" in text or ".." in text):
        parsed = float(np.mean(nums[:2]))
    else:
        parsed = nums[0]
    return parsed if parsed >= 0 else np.nan


def choose_target_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Preferred target column '{preferred}' is not in the dataset.")
        return preferred

    col = find_first_column(df.columns, TARGET_CANDIDATES)
    if col is None:
        raise ValueError(
            "Could not infer a popularity target. Pass --target-column with a column like "
            "estimated_owners, total_reviews, recommendations, or peak_ccu."
        )
    return col


def add_popularity_target(
    df: pd.DataFrame,
    target_column: str | None = None,
    quantile: float = 0.75,
    min_year_rows: int = 50,
) -> tuple[pd.DataFrame, TargetInfo]:
    """Create a within-release-year top-quantile binary popularity label."""
    out = df.copy()
    raw_col = choose_target_column(out, target_column)
    out["_popularity_value"] = out[raw_col].map(parse_numeric_popularity)
    out["_release_year"] = parse_release_year(out)

    valid = out["_popularity_value"].notna() & out["_release_year"].notna()
    out = out.loc[valid].copy()

    year_counts = out["_release_year"].value_counts()
    keep_years = year_counts[year_counts >= min_year_rows].index
    out = out[out["_release_year"].isin(keep_years)].copy()
    if out.empty:
        raise ValueError("No rows remain after target parsing and release-year filtering.")

    thresholds = out.groupby("_release_year")["_popularity_value"].transform(
        lambda s: s.quantile(quantile)
    )
    out["popular"] = (out["_popularity_value"] >= thresholds).astype(int)
    out["_release_year"] = out["_release_year"].astype(int)

    info = TargetInfo(
        target_column="popular",
        raw_target_column=raw_col,
        release_year_column="_release_year",
        threshold_quantile=quantile,
        positive_rate=float(out["popular"].mean()),
        rows=int(len(out)),
    )
    return out, info
