
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from transformers import pipeline
from detoxify import Detoxify


# =========================
# LOAD NLP MODELS
# =========================

print("\nLoading transformer models...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
)

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)

toxicity_model = Detoxify("original")

print("Models loaded successfully.\n")


# =========================
# PHARMA KEYWORDS
# =========================

PHARMA_RISK_KEYWORDS = [
    # Safety signal terms
    "death", "died", "killed",
    "side effect", "side effects",
    "adverse event", "adverse reaction", "adverse effects",
    "hospitalized", "hospitalisation",
    "lawsuit", "litigation", "class action",
    "recall", "market withdrawal",
    "toxicity", "toxic",
    "injury", "injured",
    "dangerous", "unsafe", "hazardous",
    "misinformation", "fake cure", "false claim",
    "off label", "off-label", "unapproved use",
    "contraindication", "drug interaction",
    "overdose", "black box warning",
    # Organon product + risk context
    "nexplanon side", "nuvaring risk", "implanon complication",
    "dupixent side", "rinvoq risk", "humira side",
    "biosimilar safety", "trastuzumab toxicity",
    "singulair mental", "montelukast side",
    "finasteride side", "propecia side",
    # Regulatory signals
    "FDA warning", "EMA warning", "regulatory action",
    "safety alert", "pharmacovigilance",
    "post market surveillance",
]


# =========================
# UTILITY FUNCTIONS
# =========================

def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_emojis(text):

    if not isinstance(text, str):
        return 0

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )

    return len(
        emoji_pattern.findall(text)
    )


def keyword_score(text, keywords):

    if not isinstance(text, str):
        return 0

    text = text.lower()

    return sum(
        keyword in text
        for keyword in keywords
    )


def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    print(f"\nReading file:\n{path}")

    return pd.read_csv(
        path,
        low_memory=False,
    )


# =========================
# CONTEXTUAL NLP
# =========================

def detect_sentiment(text):

    try:

        result = sentiment_pipeline(
            text[:512]
        )[0]

        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        }

        label = label_map.get(
            result["label"],
            result["label"],
        )

        return (
            label,
            round(result["score"], 4),
        )

    except Exception:

        return (
            "unknown",
            0.0,
        )


def detect_emotion(text):

    try:

        result = emotion_pipeline(
            text[:512]
        )[0]

        return (
            result["label"],
            round(result["score"], 4),
        )

    except Exception:

        return (
            "unknown",
            0.0,
        )


def detect_toxicity(text):

    try:

        result = toxicity_model.predict(
            text[:512]
        )

        max_label = max(
            result,
            key=result.get,
        )

        max_score = result[max_label]

        if max_score < 0.5:
            return (
                "non_toxic",
                round(max_score, 4),
            )

        return (
            max_label,
            round(max_score, 4),
        )

    except Exception:

        return (
            "unknown",
            0.0,
        )


# =========================
# FEATURE ENGINEERING
# =========================

def build_temporal_features(df):

    if "published_at" not in df.columns:
        df["published_at"] = (
            pd.Timestamp.utcnow()
        )

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    current_time = pd.Timestamp.utcnow()

    df["hours_since_publish"] = (
        current_time - df["published_at"]
    ).dt.total_seconds() / 3600

    df["days_since_publish"] = (
        df["hours_since_publish"] / 24
    )

    df["publish_hour"] = (
        df["published_at"].dt.hour
    )

    df["publish_weekday"] = (
        df["published_at"]
        .dt.day_name()
    )

    df["publish_month"] = (
        df["published_at"].dt.month
    )

    df["is_weekend"] = (
        df["published_at"]
        .dt.weekday >= 5
    ).astype(int)

    return df


def build_engagement_features(df):

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            df[column] = 0

        df[column] = safe_numeric(
            df[column]
        )

    df["engagement_total"] = (
        df["like_count"] +
        df["comment_count"]
    )

    df["engagement_rate"] = np.where(
        df["view_count"] > 0,
        (
            df["engagement_total"] /
            df["view_count"]
        ),
        0,
    )

    df["views_per_hour"] = np.where(
        df["hours_since_publish"] > 0,
        (
            df["view_count"] /
            df["hours_since_publish"]
        ),
        0,
    )

    df["likes_per_hour"] = np.where(
        df["hours_since_publish"] > 0,
        (
            df["like_count"] /
            df["hours_since_publish"]
        ),
        0,
    )

    df["comments_per_hour"] = np.where(
        df["hours_since_publish"] > 0,
        (
            df["comment_count"] /
            df["hours_since_publish"]
        ),
        0,
    )

    return df


def build_content_features(df):

    text_column = "analysis_text_en"

    if text_column not in df.columns:
        df[text_column] = ""

    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
    )

    df["analysis_text_clean"] = (
        df[text_column]
        .apply(normalize_text)
    )

    df["text_length"] = (
        df["analysis_text_clean"]
        .str.len()
    )

    df["word_count"] = (
        df["analysis_text_clean"]
        .str.split()
        .str.len()
    )

    df["emoji_count"] = (
        df["analysis_text_clean"]
        .apply(count_emojis)
    )

    df["question_mark_count"] = (
        df["analysis_text_clean"]
        .str.count(r"\?")
    )

    df["exclamation_count"] = (
        df["analysis_text_clean"]
        .str.count(r"!")
    )

    return df


def build_contextual_nlp_features(df):

    sentiments = []
    sentiment_scores = []

    emotions = []
    emotion_scores = []

    toxicities = []
    toxicity_scores = []

    pharma_risk_scores = []
    pharma_risk_labels = []

    total_rows = len(df)

    for index, text in enumerate(
        df["analysis_text_clean"],
        start=1,
    ):

        print(
            f"Processing NLP row "
            f"{index}/{total_rows}"
        )

        # SENTIMENT
        sentiment_label, sentiment_score = (
            detect_sentiment(text)
        )

        sentiments.append(
            sentiment_label
        )

        sentiment_scores.append(
            sentiment_score
        )

        # EMOTION
        emotion_label, emotion_score = (
            detect_emotion(text)
        )

        emotions.append(
            emotion_label
        )

        emotion_scores.append(
            emotion_score
        )

        # TOXICITY
        toxicity_label, toxicity_score = (
            detect_toxicity(text)
        )

        toxicities.append(
            toxicity_label
        )

        toxicity_scores.append(
            toxicity_score
        )

        # PHARMA RISK
        pharma_score = keyword_score(
            text,
            PHARMA_RISK_KEYWORDS,
        )

        pharma_risk_scores.append(
            pharma_score
        )

        pharma_risk_labels.append(
            "high_risk"
            if pharma_score > 0
            else "low_risk"
        )

    df["sentiment_label"] = sentiments
    df["sentiment_score"] = sentiment_scores

    df["emotion_label"] = emotions
    df["emotion_score"] = emotion_scores

    df["toxicity_label"] = toxicities
    df["toxicity_score"] = toxicity_scores

    df["pharma_risk_score"] = (
        pharma_risk_scores
    )

    df["pharma_risk_label"] = (
        pharma_risk_labels
    )

    return df


def build_virality_features(df):

    df["virality_score"] = (
        (
            df["engagement_rate"] * 0.4
        ) +
        (
            df["views_per_hour"] * 0.3
        ) +
        (
            df["comments_per_hour"] * 0.2
        ) +
        (
            df["sentiment_score"] * 0.1
        )
    )

    threshold = (
        df["virality_score"]
        .quantile(0.95)
    )

    df["viral_label"] = (
        df["virality_score"] >= threshold
    ).astype(int)

    return df


def build_feature_pipeline(df):

    print(
        "\nBuilding temporal features..."
    )

    df = build_temporal_features(df)

    print(
        "Building engagement features..."
    )

    df = build_engagement_features(df)

    print(
        "Building content features..."
    )

    df = build_content_features(df)

    print(
        "Building contextual NLP features..."
    )

    df = build_contextual_nlp_features(df)

    print(
        "Building virality features..."
    )

    df = build_virality_features(df)

    return df


# =========================
# OUTPUT
# =========================

def save_output(df, output_path, incremental=False):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_path.with_suffix(
            ".parquet"
        )
    )

    csv_path = (
        output_path.with_suffix(".csv")
    )

    if incremental and csv_path.exists():

        df.to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nAppended {len(df)} rows to CSV:\n"
            f"{csv_path}"
        )

        try:

            full_df = pd.read_csv(
                csv_path,
                low_memory=False,
            )

            full_df.to_parquet(
                parquet_path,
                index=False,
            )

            print(
                f"\nRebuilt parquet:\n"
                f"{parquet_path}"
            )

        except Exception as error:

            print(
                "\nParquet rebuild failed:"
            )

            print(str(error))

        return

    try:

        df.to_parquet(
            parquet_path,
            index=False,
        )

        print(
            f"\nSaved parquet:\n"
            f"{parquet_path}"
        )

    except Exception as error:

        print(
            "\nParquet save failed:"
        )

        print(str(error))

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved CSV:\n"
        f"{csv_path}"
    )


def default_output_path(input_path):

    return input_path.with_name(
        f"{input_path.stem}_features"
    )


def find_cleaned_file():

    current_dir = Path.cwd()

    cleaned_files = list(
        current_dir.glob("*_cleaned.csv")
    )

    if not cleaned_files:
        raise FileNotFoundError(
            "No cleaned CSV found."
        )

    latest_file = max(
        cleaned_files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    return latest_file


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Contextual YouTube "
            "Feature Engineering"
        )
    )

    parser.add_argument(
        "--input",
        help="Optional input CSV",
    )

    parser.add_argument(
        "--output",
        help="Optional output path",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Process only new video_ids not already in the output file.",
    )

    args = parser.parse_args()

    if args.input:

        input_path = Path(args.input)

    else:

        print(
            "\nSearching for latest "
            "*_cleaned.csv..."
        )

        input_path = find_cleaned_file()

    print(
        f"\nUsing file:\n"
        f"{input_path}"
    )

    df = load_dataset(input_path)

    print(
        f"\nLoaded rows: {len(df)}"
    )

    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(
            input_path
        )
    )

    if args.incremental:

        csv_output = output_path.with_suffix(".csv")

        if csv_output.exists():

            existing_df = pd.read_csv(
                csv_output,
                low_memory=False,
                usecols=["video_id"],
            )

            processed_ids = set(
                existing_df["video_id"]
                .dropna()
                .astype(str)
            )

            df = df[
                ~df["video_id"]
                .astype(str)
                .isin(processed_ids)
            ].copy()

            print(
                f"\nIncremental: {len(df)} new rows "
                f"(skipped {len(processed_ids)} already processed)"
            )

        else:

            print(
                "\nIncremental: no existing output, processing all rows"
            )

    if len(df) == 0:

        print(
            "\nNo new rows to process."
        )

        return

    feature_df = build_feature_pipeline(df)

    save_output(
        feature_df,
        output_path,
        incremental=args.incremental,
    )

    print(
        "\nFeature engineering "
        "completed successfully."
    )


if __name__ == "__main__":
    main()

 