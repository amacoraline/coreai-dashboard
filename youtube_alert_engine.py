import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# CONFIG
# ==========================================

ALERT_OUTPUT_FILE = (
    "youtube_alerts.csv"
)


# ==========================================
# SCORE RANGE THRESHOLDS
# ==========================================

SCORE_RANGE_THRESHOLDS = {
    # 0–1 probability / fraction scores
    "viral_probability":          (0.40, 0.70),
    "toxicity_score":             (0.30, 0.60),
    "sentiment_score":            (0.40, 0.70),
    "emotion_score":              (0.40, 0.70),
    "narrative_similarity_score": (0.40, 0.70),
    "ravs_score":                 (0.35, 0.70),
    "alert_score":                (0.35, 0.70),
    "virality_score":             (0.40, 0.70),
    # Engagement rate (fraction, 0–1+)
    "engagement_rate":            (0.02, 0.05),
    # Count-based
    "pharma_risk_score":          (1,    3   ),
    "cluster_volume":             (10,   30  ),
    # Per-hour rates
    "views_per_hour":             (100,  1000),
    "likes_per_hour":             (10,   100 ),
    "comments_per_hour":          (5,    50  ),
    # Velocity (change per hour between snapshots)
    "view_velocity":              (100,  1000),
    "like_velocity":              (10,   100 ),
    "comment_velocity":           (5,    50  ),
    "growth_acceleration":        (0,    500 ),
    # Trend / narrative scores
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

TOP_ALERT_COUNT = 100

VIRALITY_THRESHOLD = 0.80

TOXICITY_THRESHOLD = 0.75

PHARMA_RISK_THRESHOLD = 1

TREND_PERCENTILE = 0.95

SPIKE_PERCENTILE = 0.95


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def load_dataset(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    print(
        f"\nLoading dataset:\n{path}"
    )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def find_latest_prediction_file():

    current_dir = Path.cwd()

    files = list(
        current_dir.glob(
            "youtube_virality_predictions.csv"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No youtube_virality_predictions.csv found."
        )

    latest_file = max(
        files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    return latest_file


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


# ==========================================
# PREPROCESS
# ==========================================

def preprocess(df):

    numeric_columns = [

        "viral_probability",
        "toxicity_score",
        "pharma_risk_score",
        "trend_score",
        "growth_acceleration",
        "view_velocity",
        "engagement_rate",
        "narrative_trend_score",
        "cluster_volume",
    ]

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0

        df[column] = safe_numeric(
            df[column]
        )

    return df


# ==========================================
# RAVS SCORE
# ==========================================

def compute_ravs_score(df):
    """
    Risk-Adjusted Virality Score (RAVS).
    Combines viral probability, toxicity, engagement trend,
    pharma risk keyword density, and influencer amplification.
    """

    def norm(series):
        lo = series.min()
        hi = series.max()
        if hi == lo:
            return series * 0.0
        return (series - lo) / (hi - lo)

    viral = (
        df["viral_probability"].fillna(0)
        if "viral_probability" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    toxicity = (
        df["toxicity_score"].fillna(0)
        if "toxicity_score" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    trend = norm(
        df["trend_score"].fillna(0)
        if "trend_score" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    pharma = norm(
        df["pharma_risk_score"].fillna(0)
        if "pharma_risk_score" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    sub_col = (
        "channel_subscriber_count"
        if "channel_subscriber_count" in df.columns
        else "subscriber_count"
    )
    sub_count = safe_numeric(
        df[sub_col]
        if sub_col in df.columns
        else pd.Series(0, index=df.index)
    )
    influencer = (sub_count >= 100_000).astype(float)

    df["ravs_score"] = (
        viral * 0.30
        + toxicity * 0.20
        + trend * 0.20
        + pharma * 0.20
        + influencer * 0.10
    ).round(4)

    # Sub-component scores (0-100 scale) for dashboard display
    vol_col = (
        df["cluster_volume"].fillna(0)
        if "cluster_volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    df["ravs_speed_of_engagement"] = (
        (viral * 100)
        .clip(40, 100)
        .astype(int)
    )

    df["ravs_regulatory_relevance"] = np.where(
        df["pharma_risk_score"].fillna(0) > 0,
        (df["pharma_risk_score"].fillna(0) / 5 * 82).clip(35, 100).astype(int),
        (viral * 75).clip(35, 100).astype(int),
    )

    df["ravs_safety_implications"] = (
        (df["toxicity_score"].fillna(0) * 100)
        .clip(20, 100)
        .astype(int)
    )

    df["ravs_influencer_reach"] = np.where(
        sub_count > 0,
        (sub_count / 500_000 * 61).clip(25, 100).astype(int),
        35,
    )

    df["ravs_topic_sensitivity"] = np.where(
        vol_col > 0,
        (vol_col / 50 * 59).clip(25, 100).astype(int),
        (viral * 60).clip(25, 100).astype(int),
    )

    return df


# ==========================================
# ALERT RULES
# ==========================================

def detect_viral_alerts(df):

    print(
        "\nDetecting viral alerts..."
    )

    viral_df = df[
        df["viral_probability"]
        >= VIRALITY_THRESHOLD
    ].copy()

    viral_df["alert_type"] = (
        "viral_emerging"
    )

    viral_df["alert_priority"] = (
        "high"
    )

    viral_df["alert_reason"] = (
        "High virality probability detected"
    )

    return viral_df


def detect_toxicity_alerts(df):

    print(
        "\nDetecting toxicity alerts..."
    )

    toxic_df = df[
        df["toxicity_score"]
        >= TOXICITY_THRESHOLD
    ].copy()

    toxic_df["alert_type"] = (
        "toxicity_spike"
    )

    toxic_df["alert_priority"] = (
        "high"
    )

    toxic_df["alert_reason"] = (
        "High toxicity content detected"
    )

    return toxic_df


def detect_pharma_risk_alerts(df):

    print(
        "\nDetecting pharma risk alerts..."
    )

    risk_df = df[
        df["pharma_risk_score"]
        >= PHARMA_RISK_THRESHOLD
    ].copy()

    risk_df["alert_type"] = (
        "pharma_risk"
    )

    risk_df["alert_priority"] = (
        "critical"
    )

    risk_df["alert_reason"] = (
        "Potential pharma safety risk detected"
    )

    return risk_df


def detect_trending_narratives(df):

    print(
        "\nDetecting emerging narratives..."
    )

    threshold = (
        df["narrative_trend_score"]
        .quantile(
            TREND_PERCENTILE
        )
    )

    trend_df = df[
        df["narrative_trend_score"]
        >= threshold
    ].copy()

    trend_df["alert_type"] = (
        "emerging_narrative"
    )

    trend_df["alert_priority"] = (
        "medium"
    )

    trend_df["alert_reason"] = (
        "Narrative cluster accelerating rapidly"
    )

    return trend_df


def detect_growth_spikes(df):

    print(
        "\nDetecting growth spikes..."
    )

    threshold = (
        df["growth_acceleration"]
        .quantile(
            SPIKE_PERCENTILE
        )
    )

    spike_df = df[
        df["growth_acceleration"]
        >= threshold
    ].copy()

    spike_df["alert_type"] = (
        "growth_spike"
    )

    spike_df["alert_priority"] = (
        "medium"
    )

    spike_df["alert_reason"] = (
        "Rapid growth acceleration detected"
    )

    return spike_df


def detect_influencer_alerts(df):

    print(
        "\nDetecting influencer amplification..."
    )

    if (
        "subscriber_count"
        not in df.columns
    ):

        df["subscriber_count"] = 0

    influencer_df = df[
        (
            df["subscriber_count"]
            >= 100000
        )
        &
        (
            df["trend_score"]
            >= df[
                "trend_score"
            ].quantile(0.90)
        )
    ].copy()

    influencer_df["alert_type"] = (
        "influencer_amplification"
    )

    influencer_df["alert_priority"] = (
        "high"
    )

    influencer_df["alert_reason"] = (
        "High influence channel accelerating narrative spread"
    )

    return influencer_df


# ==========================================
# BUILD ALERT SCORE
# ==========================================

def build_alert_score(df):

    df["alert_score"] = (

        (
            df["viral_probability"]
            .fillna(0)
            * 0.30
        )

        +

        (
            df["toxicity_score"]
            .fillna(0)
            * 0.20
        )

        +

        (
            df["trend_score"]
            .fillna(0)
            * 0.20
        )

        +

        (
            df["growth_acceleration"]
            .fillna(0)
            * 0.15
        )

        +

        (
            df["pharma_risk_score"]
            .fillna(0)
            * 0.15
        )
    )

    return df


# ==========================================
# DEDUP ALERTS
# ==========================================

def deduplicate_alerts(df):

    sort_columns = [
        "alert_score"
    ]

    df = df.sort_values(
        by=sort_columns,
        ascending=False,
    )

    dedup_columns = [
        "video_id",
        "alert_type",
    ]

    df = df.drop_duplicates(
        subset=dedup_columns
    )

    return df


# ==========================================
# SELECT FINAL COLUMNS
# ==========================================

def finalize_alerts(df):

    important_columns = [

        "video_id",
        "title",
        "channel_title",

        "alert_type",
        "alert_priority",
        "alert_reason",
        "ravs_speed_of_engagement",
        "ravs_regulatory_relevance",
        "ravs_safety_implications",
        "ravs_influencer_reach",
        "ravs_topic_sensitivity",

        "viral_probability",
        "viral_probability_range",
        "trend_score",
        "trend_score_range",
        "growth_acceleration",
        "growth_acceleration_range",

        "sentiment_label",
        "emotion_label",
        "toxicity_label",

        "toxicity_score",
        "toxicity_score_range",
        "pharma_risk_score",
        "pharma_risk_score_range",

        "engagement_rate",
        "engagement_rate_range",
        "views_per_hour",
        "views_per_hour_range",
        "view_velocity",
        "view_velocity_range",

        "narrative_cluster_id",
        "dominant_topic_id",
        "topic_keywords",
        "cluster_topic_name",

        "cluster_volume",
        "cluster_volume_range",
        "narrative_trend_score",
        "narrative_trend_score_range",

        "ravs_score",
        "ravs_score_range",
        "alert_score",
        "alert_score_range",

        "view_count",
        "like_count",
        "comment_count",

        "subscriber_count",

        "tracking_timestamp",
    ]

    existing_columns = [
        column
        for column in important_columns
        if column in df.columns
    ]

    return df[
        existing_columns
    ]


# ==========================================
# MAIN ALERT PIPELINE
# ==========================================

def build_alert_pipeline(df):

    df = preprocess(df)
    df = compute_ravs_score(df)
    df = add_score_range_labels(df)

    alert_frames = []

    # VIRAL
    viral_alerts = (
        detect_viral_alerts(df)
    )

    alert_frames.append(
        viral_alerts
    )

    # TOXICITY
    toxicity_alerts = (
        detect_toxicity_alerts(df)
    )

    alert_frames.append(
        toxicity_alerts
    )

    # PHARMA RISK
    pharma_alerts = (
        detect_pharma_risk_alerts(df)
    )

    alert_frames.append(
        pharma_alerts
    )

    # NARRATIVES
    narrative_alerts = (
        detect_trending_narratives(df)
    )

    alert_frames.append(
        narrative_alerts
    )

    # GROWTH SPIKES
    spike_alerts = (
        detect_growth_spikes(df)
    )

    alert_frames.append(
        spike_alerts
    )

    # INFLUENCERS
    influencer_alerts = (
        detect_influencer_alerts(df)
    )

    alert_frames.append(
        influencer_alerts
    )

    # MERGE
    alerts_df = pd.concat(
        alert_frames,
        ignore_index=True,
    )

    if len(alerts_df) == 0:

        print(
            "\nNo alerts generated."
        )

        return alerts_df

    # SCORE
    alerts_df = build_alert_score(
        alerts_df
    )

    # DEDUP
    alerts_df = deduplicate_alerts(
        alerts_df
    )

    # SORT
    alerts_df = alerts_df.sort_values(
        by="alert_score",
        ascending=False,
    )

    # LIMIT
    alerts_df = alerts_df.head(
        TOP_ALERT_COUNT
    )

    # FINALIZE
    alerts_df = finalize_alerts(
        alerts_df
    )

    return alerts_df


# ==========================================
# SAVE OUTPUTS
# ==========================================

def save_outputs(df):

    if len(df) == 0:

        print(
            "\nNo alerts generated — writing empty CSV."
        )

        # Write header-only CSV so the stage file always exists
        pd.DataFrame(
            columns=[
                "video_id", "title", "channel_title",
                "alert_type", "alert_priority", "alert_reason",
                "alert_score", "ravs_score", "viral_probability",
            ]
        ).to_csv(
            ALERT_OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        return

    # CSV
    df.to_csv(
        ALERT_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved alerts CSV:\n"
        f"{ALERT_OUTPUT_FILE}"
    )

    # PARQUET
    try:

        df.to_parquet(
            "youtube_alerts.parquet",
            index=False,
        )

        print(
            "\nSaved alerts parquet."
        )

    except Exception as error:

        print(
            "\nParquet save failed:"
        )

        print(str(error))


# ==========================================
# SUMMARY
# ==========================================

def print_summary(df):

    if len(df) == 0:

        return

    print(
        "\n========== ALERT SUMMARY =========="
    )

    print(
        f"\nTotal alerts: {len(df)}"
    )

    print(
        "\nAlert types:"
    )

    print(
        df["alert_type"]
        .value_counts()
    )

    print(
        "\nPriority distribution:"
    )

    print(
        df["alert_priority"]
        .value_counts()
    )

    print(
        "\nTop narratives:"
    )

    if (
        "topic_keywords"
        in df.columns
    ):

        print(
            df["topic_keywords"]
            .value_counts()
            .head(10)
        )


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YouTube Alert Engine"
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional virality prediction CSV"
        ),
    )

    args = parser.parse_args()

    # FIND FILE
    if args.input:

        input_path = Path(
            args.input
        )

    else:

        print(
            "\nSearching for "
            "youtube_virality_predictions.csv..."
        )

        input_path = (
            find_latest_prediction_file()
        )

    print(
        f"\nUsing file:\n"
        f"{input_path}"
    )

    # LOAD
    df = load_dataset(
        input_path
    )

    print(
        f"\nLoaded rows: {len(df)}"
    )

    # BUILD ALERTS
    alerts_df = (
        build_alert_pipeline(
            df
        )
    )

    # SAVE
    save_outputs(
        alerts_df
    )

    # SUMMARY
    print_summary(
        alerts_df
    )

    print(
        "\nAlert engine completed successfully."
    )


if __name__ == "__main__":
    main()