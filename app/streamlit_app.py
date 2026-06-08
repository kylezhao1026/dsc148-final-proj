from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_experiment_model.joblib")
FALLBACK_MODEL_PATH = Path("models/best_model.joblib")
METADATA_PATH = Path("reports/tables/experiment_metadata.json")
FALLBACK_METADATA_PATH = Path("reports/tables/run_metadata.json")

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #18202a;
            --muted: #667085;
            --line: #d8dee8;
            --panel: #ffffff;
            --soft: #f4f7fb;
            --accent: #10a37f;
            --accent-2: #e85d3f;
        }

        .stApp {
            background:
                linear-gradient(180deg, #eef3f8 0%, #f8fafc 38%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.1rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, label {
            color: var(--ink) !important;
            letter-spacing: 0 !important;
        }

        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.86);
            backdrop-filter: blur(10px);
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff 0%, #f3f7fb 62%, #eaf6f2 100%);
            border-radius: 8px;
            padding: 1.6rem 1.7rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 12px 32px rgba(24, 32, 42, 0.07);
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.1rem);
            font-weight: 780;
            line-height: 1.02;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.75rem 0 0;
        }

        .stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.95rem 0 1.1rem;
        }

        .stat {
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .stat-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 760;
            margin-top: 0.15rem;
        }

        .section-title {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 760;
            margin: 0.35rem 0 0.45rem;
        }

        .result {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 10px 24px rgba(24, 32, 42, 0.07);
        }

        .result-score {
            font-size: clamp(2.4rem, 8vw, 4.8rem);
            line-height: 0.95;
            color: var(--accent);
            font-weight: 820;
            margin: 0.2rem 0 0.6rem;
        }

        .result-label {
            color: var(--muted);
            font-weight: 650;
        }

        .result-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 10px 24px rgba(24, 32, 42, 0.06);
        }

        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 0;
            background: var(--ink);
            color: #ffffff;
            font-weight: 750;
            padding: 0.75rem 1rem;
        }

        .stButton > button:hover {
            background: #263241;
            color: #ffffff;
            border: 0;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: #cbd5e1 !important;
        }

        @media (max-width: 800px) {
            .stat-row {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if FALLBACK_METADATA_PATH.exists():
        return json.loads(FALLBACK_METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def expected_feature_columns(metadata: dict) -> list[str]:
    feature_spec = metadata.get("feature_spec", {})
    return (
        feature_spec.get("numeric_columns", [])
        + feature_spec.get("categorical_columns", [])
        + feature_spec.get("text_columns", [])
    )


def build_input_row(values: dict, metadata: dict) -> pd.DataFrame:
    feature_spec = metadata.get("feature_spec", {})
    row = pd.DataFrame([values])
    for column in expected_feature_columns(metadata):
        if column not in row.columns:
            row[column] = 0 if column in feature_spec.get("numeric_columns", []) else ""
    return row


st.set_page_config(page_title="Steam Popularity Predictor", layout="wide")
inject_css()

model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH

if not model_path.exists():
    st.warning("Train a model first with `python -m src.train --data data/games_march2025_cleaned.csv`.")
    st.stop()

model = joblib.load(model_path)
metadata = load_metadata()
target = metadata.get("target", {})

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">DSC 148 final project demo</div>
        <h1 class="hero-title">Steam Popularity Predictor</h1>
        <p class="hero-copy">
            Estimate whether a Steam game lands in the top quartile of review activity
            for its release-year cohort using launch-visible store metadata.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

train_years = metadata.get("train_years", [])
test_years = metadata.get("test_years", [])
st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat">
            <div class="stat-label">Training rows</div>
            <div class="stat-value">{metadata.get("train_rows", "n/a")}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Target</div>
            <div class="stat-value">Top {int((1 - target.get("threshold_quantile", 0.75)) * 100)}%</div>
        </div>
        <div class="stat">
            <div class="stat-label">Held-out years</div>
            <div class="stat-value">{test_years[0] if test_years else "n/a"}-{test_years[-1] if test_years else "n/a"}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

default_probability: float | None = None

with st.form("game_form", clear_on_submit=False):
    st.markdown('<div class="section-title">Game profile</div>', unsafe_allow_html=True)
    left, middle, right = st.columns([1.1, 1, 1])

    with left:
        name = st.text_input("Title", "Example Quest")
        description = st.text_area(
            "Short description",
            "A narrative-driven adventure game with exploration, combat, and puzzle solving.",
            height=144,
        )

    with middle:
        genres = st.text_input("Genres", "Action, Adventure")
        tags = st.text_input("Tags", "Singleplayer, Indie, Story Rich")
        categories = st.text_input("Categories", "Single-player, Steam Achievements")
        supported_languages = st.text_input("Languages", "English, Spanish")

    with right:
        price = st.number_input("Price", min_value=0.0, value=19.99, step=1.0)
        release_month_name = st.selectbox("Release month", list(MONTHS.keys()), index=9)
        required_age = st.selectbox("Age rating", [0, 3, 7, 12, 16, 18], index=0)
        dlc_count = st.number_input("DLC count", min_value=0, value=0, step=1)
        discount = st.slider("Launch discount", 0, 95, 0, step=5)

    st.markdown('<div class="section-title">Platforms</div>', unsafe_allow_html=True)
    platform_cols = st.columns(3)
    with platform_cols[0]:
        windows = st.checkbox("Windows", value=True)
    with platform_cols[1]:
        mac = st.checkbox("macOS", value=False)
    with platform_cols[2]:
        linux = st.checkbox("Linux", value=False)

    submitted = st.form_submit_button("Predict popularity")

if submitted:
    row = build_input_row(
        {
            "name": name,
            "price": price,
            "windows": int(windows),
            "mac": int(mac),
            "linux": int(linux),
            "dlc_count": dlc_count,
            "discount": discount,
            "release_month": MONTHS[release_month_name],
            "required_age": required_age,
            "genres": genres,
            "tags": tags,
            "categories": categories,
            "supported_languages": supported_languages,
            "developers": "",
            "publishers": "",
            "detailed_description": description,
            "about_the_game": description,
            "short_description": description,
        },
        metadata,
    )
    default_probability = float(model.predict_proba(row)[0, 1])

if default_probability is not None:
    label = "Strong candidate" if default_probability >= 0.5 else "Long-shot candidate"
    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">{label}</div>
            <div class="result-score">{default_probability:.1%}</div>
            <div class="result-note">
                Probability of top-quartile review activity among games released in the same year.
                Current model: {metadata.get("best_experiment", metadata.get("best_model", "trained baseline"))}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="result">
            <div class="result-label">Prediction</div>
            <div class="result-score">--</div>
            <div class="result-note">Fill out the game profile and run the model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
