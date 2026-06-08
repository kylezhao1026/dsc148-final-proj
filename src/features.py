from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.load_data import find_first_column, parse_release_month


LEAKY_SUBSTRINGS = [
    "owner",
    "review",
    "recommendation",
    "ccu",
    "player",
    "playtime",
    "rating",
    "score",
    "positive",
    "negative",
    "pct_pos",
    "metacritic",
    "achievement",
    "rank",
    "revenue",
    "sale",
    "followers",
]

RAW_DATE_SUBSTRINGS = [
    "release_date",
    "date_release",
    "released",
]

DROP_SUBSTRINGS = [
    "appid",
    "steam_appid",
    "url",
    "email",
    "image",
    "screenshot",
    "movie",
    "package",
]

TEXT_CANDIDATES = [
    "short_description",
    "detailed_description",
    "about_the_game",
    "description",
    "name",
]

TAG_CANDIDATES = [
    "genres",
    "genre",
    "tags",
    "categories",
    "supported_languages",
    "developers",
    "publishers",
]

CATEGORICAL_CANDIDATES = [
    "type",
    "required_age",
    "windows",
    "mac",
    "linux",
    "is_free",
    "controller_support",
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric_columns: list[str]
    categorical_columns: list[str]
    text_columns: list[str]
    dropped_columns: list[str]


FEATURE_GROUPS = {
    "numeric": {"required_age", "price", "dlc_count", "windows", "mac", "linux", "discount", "release_month"},
    "descriptions": {"short_description", "detailed_description", "about_the_game", "description", "name"},
    "store_metadata": {
        "genres",
        "genre",
        "tags",
        "categories",
        "supported_languages",
        "developers",
        "publishers",
    },
}


def _is_leaky_column(col: str, raw_target_column: str) -> bool:
    if col in {"popular", "_popularity_value"}:
        return True
    if col == raw_target_column:
        return True
    if any(part in col for part in DROP_SUBSTRINGS):
        return True
    if col != "release_month" and any(part in col for part in RAW_DATE_SUBSTRINGS):
        return True
    return any(part in col for part in LEAKY_SUBSTRINGS)


def _existing(columns: pd.Index, candidates: list[str]) -> list[str]:
    found: list[str] = []
    for candidate in candidates:
        col = find_first_column(columns, [candidate])
        if col and col not in found:
            found.append(col)
    return found


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["release_month"] = parse_release_month(out)

    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype(int)

    return out


def infer_feature_spec(df: pd.DataFrame, raw_target_column: str) -> FeatureSpec:
    ignored = {
        "popular",
        "_popularity_value",
        "_release_year",
    }
    dropped = sorted(
        col for col in df.columns if col in ignored or _is_leaky_column(col, raw_target_column)
    )

    usable = [col for col in df.columns if col not in dropped]
    numeric = [
        col
        for col in usable
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 1
    ]

    text_cols = [col for col in _existing(df.columns, TEXT_CANDIDATES + TAG_CANDIDATES) if col in usable]
    categorical = [
        col
        for col in _existing(df.columns, CATEGORICAL_CANDIDATES)
        if col in usable and col not in text_cols and not pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in usable:
        if col in numeric or col in text_cols or col in categorical:
            continue
        if df[col].dtype == object and 2 <= df[col].nunique(dropna=True) <= 50:
            categorical.append(col)

    return FeatureSpec(
        numeric_columns=numeric,
        categorical_columns=categorical,
        text_columns=text_cols,
        dropped_columns=dropped,
    )


def combine_text_columns(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame.fillna("").astype(str).agg(" ".join, axis=1)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []

    if spec.numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                spec.numeric_columns,
            )
        )

    if spec.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                spec.categorical_columns,
            )
        )

    if spec.text_columns:
        transformers.append(
            (
                "text",
                Pipeline(
                    steps=[
                        (
                            "combine",
                            FunctionTransformer(combine_text_columns, validate=False),
                        ),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=12000,
                                min_df=3,
                                ngram_range=(1, 2),
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
                spec.text_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were inferred.")

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def select_feature_groups(spec: FeatureSpec, groups: set[str]) -> FeatureSpec:
    """Return a FeatureSpec restricted to high-level feature groups."""
    allowed: set[str] = set()
    for group in groups:
        allowed.update(FEATURE_GROUPS[group])

    numeric = [col for col in spec.numeric_columns if col in allowed]
    categorical = [col for col in spec.categorical_columns if col in allowed]
    text = [col for col in spec.text_columns if col in allowed]
    dropped = sorted(set(spec.dropped_columns) | (set(spec.numeric_columns) - set(numeric)))
    dropped = sorted(set(dropped) | (set(spec.categorical_columns) - set(categorical)))
    dropped = sorted(set(dropped) | (set(spec.text_columns) - set(text)))
    return FeatureSpec(
        numeric_columns=numeric,
        categorical_columns=categorical,
        text_columns=text,
        dropped_columns=dropped,
    )
