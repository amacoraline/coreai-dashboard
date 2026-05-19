"""
Virality prediction — inference only.

Loads youtube_virality_model.pkl and generates predictions on the latest
narrative detection data. Run youtube_virality_model.py first to train.

Usage:
    python youtube_virality_predict.py
    python youtube_virality_predict.py --input youtube_narrative_detection.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_FILE = "youtube_virality_model.pkl"
PREDICTION_FILE = "youtube_virality_predictions.csv"

SCORE_RANGE_THRESHOLDS = {
    "viral_probability":          (0.40, 0.70),
    "toxicity_score":             (0.30, 0.60),
    "sentiment_score":            (0.40, 0.70),
    "emotion_score":              (0.40, 0.70),
    "narrative_similarity_score": (0.40, 0.70),
    "virality_score":             (0.40, 0.70),
    "engagement_rate":            (0.02, 0.05),
    "pharma_risk_score":          (1,    3   ),
    "cluster_volume":             (10,   30  ),
    "views_per_hour":             (100,  1000),
    "likes_per_hour":             (10,   100 ),
    "comments_per_hour":          (5,    50  ),
    "view_velocity":              (100,  1000),
    "like_velocity":              (10,   100 ),
    "comment_velocity":           (5,    50  ),
    "growth_acceleration":        (0,    500 ),
    "trend_score":                (0.30, 0.70),
    "narrative_trend_score":      (5,    15  ),
}


def add_score_range_labels(df):

    for col, (low_thresh, high_thresh) in (
        SCORE_RANGE_THRESHOLDS.items()
    ):

        if col not in df.columns:
            continue

        series = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

        df[f"{col}_range"] = np.select(
            [
                series < low_thresh,
                (series >= low_thresh) & (series < high_thresh),
                series >= high_thresh,
            ],
            ["Low", "Medium", "High"],
            default="Low",
        )

    return df

FEATURE_COLUMNS = [
    # Engagement velocity
    "engagement_rate",
    "views_per_hour",
    "likes_per_hour",
    "comments_per_hour",
    "view_velocity",
    "like_velocity",
    "comment_velocity",
    "growth_acceleration",
    "trend_score",
    # Content
    "text_length",
    "word_count",
    "emoji_count",
    "question_mark_count",
    "exclamation_count",
    # NLP scores
    "sentiment_score",
    "emotion_score",
    "toxicity_score",
    # Pharma risk
    "pharma_risk_score",
    # Narrative
    "narrative_similarity_score",
    "cluster_volume",
    "narrative_trend_score",
    # Categorical
    "sentiment_label",
    "emotion_label",
    "toxicity_label",
    "pharma_risk_label",
    "trend_label",
    "emerging_narrative",
]


def find_narrative_file() -> Path:
    files = list(Path.cwd().glob("youtube_narrative_detection.csv"))
    if not files:
        raise FileNotFoundError("youtube_narrative_detection.csv not found.")
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Virality prediction — inference only."
    )
    parser.add_argument(
        "--input",
        help="Narrative detection CSV (auto-detected if omitted).",
    )
    args = parser.parse_args()

    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {MODEL_FILE}\n"
            "Train first:  python youtube_virality_model.py"
        )

    print(f"\nLoading model: {MODEL_FILE}")
    pipeline = joblib.load(model_path)

    input_path = Path(args.input) if args.input else find_narrative_file()
    print(f"Loading data:  {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df)} rows")

    existing_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = sorted(set(FEATURE_COLUMNS) - set(existing_features))
    if missing:
        print(f"Missing features (pipeline handles defaults): {missing}")

    X = df[existing_features].copy()

    print("Generating predictions...")
    probabilities = pipeline.predict_proba(X)[:, 1]

    df["viral_probability"] = np.round(probabilities, 4)
    df["predicted_viral"] = (df["viral_probability"] >= 0.7).astype(int)

    df = add_score_range_labels(df)

    df.to_csv(PREDICTION_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {PREDICTION_FILE}")
    print(f"Predicted viral: {df['predicted_viral'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
