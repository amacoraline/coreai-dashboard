
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================
# CONFIG
# =====================================

TRACKING_HISTORY_FILE = (
    "youtube_tracking_history.csv"
)

TRACKING_FEATURES_FILE = (
    "youtube_tracking_features.csv"
)

SNAPSHOTS_FILE = (
    "youtube_video_snapshots.csv"
)

# Columns that come from live snapshots (change over time)
LIVE_METRIC_COLUMNS = [
    "view_count",
    "like_count",
    "comment_count",
    "channel_subscriber_count",
]


# =====================================
# UTILITY FUNCTIONS
# =====================================

def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def find_latest_feature_file():

    current_dir = Path.cwd()

    feature_files = list(
        current_dir.glob(
            "*_cleaned_features.csv"
        )
    )

    if not feature_files:

        raise FileNotFoundError(
            "No *_cleaned_features.csv file found."
        )

    latest_file = max(
        feature_files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    return latest_file


def load_feature_dataset(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    print(
        f"\nLoading feature dataset:\n{path}"
    )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def load_latest_snapshots(snapshots_path):
    """Return the most recent snapshot row per video_id from youtube_video_snapshots.csv."""

    if not snapshots_path.exists():
        print("\nNo snapshots file found — using extraction-time metrics.")
        return pd.DataFrame()

    print(f"\nLoading live snapshots:\n{snapshots_path}")

    snap_df = pd.read_csv(snapshots_path, low_memory=False)

    if snap_df.empty or "video_id" not in snap_df.columns:
        return pd.DataFrame()

    snap_df["snapshot_time"] = pd.to_datetime(
        snap_df.get("snapshot_time", pd.NaT), errors="coerce", utc=True
    )

    # Keep only the most recent snapshot per video
    latest = (
        snap_df.sort_values("snapshot_time")
        .groupby("video_id", as_index=False)
        .last()
    )

    return latest[["video_id"] + [c for c in LIVE_METRIC_COLUMNS if c in latest.columns]]


def merge_live_metrics(feature_df, snapshots_df):
    """Overwrite stale extraction-time counts with live snapshot counts."""

    if snapshots_df.empty:
        return feature_df

    # Drop stale metric columns from features before merging
    cols_to_drop = [c for c in LIVE_METRIC_COLUMNS if c in feature_df.columns]
    feature_df = feature_df.drop(columns=cols_to_drop)

    merged = feature_df.merge(snapshots_df, on="video_id", how="left")

    # Fill any videos with no snapshot with 0
    for col in LIVE_METRIC_COLUMNS:
        if col in merged.columns:
            merged[col] = safe_numeric(merged[col])

    print(f"\nMerged live metrics for {snapshots_df['video_id'].nunique()} videos.")
    return merged


# =====================================
# BUILD SNAPSHOT
# =====================================

def build_tracking_snapshot(df):

    current_timestamp = (
        pd.Timestamp.utcnow()
    )

    required_columns = [

        # IDENTIFIERS
        "video_id",
        "title",
        "channel_title",

        # TEXT DATA
        "analysis_text_en",
        "analysis_text_clean",
        "title_en",
        "description_en",
        "transcript_en",
        "comments_text_full_en",

        # METRICS
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",

        # ENGAGEMENT
        "engagement_rate",
        "virality_score",
        "viral_label",

        # NLP
        "sentiment_label",
        "sentiment_score",
        "emotion_label",
        "emotion_score",
        "toxicity_label",
        "toxicity_score",

        # PHARMA RISK
        "pharma_risk_label",
        "pharma_risk_score",
    ]

    for column in required_columns:

        if column not in df.columns:

            df[column] = np.nan

    snapshot_df = df[
        required_columns
    ].copy()

    snapshot_df[
        "tracking_timestamp"
    ] = current_timestamp

    return snapshot_df


# =====================================
# LOAD HISTORY
# =====================================

def load_tracking_history(path):

    if not path.exists():

        print(
            "\nNo tracking history found."
        )

        return pd.DataFrame()

    print(
        f"\nLoading tracking history:\n{path}"
    )

    history_df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "tracking_timestamp" in history_df.columns:

        history_df[
            "tracking_timestamp"
        ] = pd.to_datetime(
            history_df[
                "tracking_timestamp"
            ],
            errors="coerce",
            utc=True,
        )

    return history_df


# =====================================
# APPEND HISTORY
# =====================================

def append_tracking_history(
    history_df,
    snapshot_df,
):

    combined_df = pd.concat(
        [
            history_df,
            snapshot_df,
        ],
        ignore_index=True,
    )

    combined_df[
        "tracking_timestamp"
    ] = pd.to_datetime(
        combined_df[
            "tracking_timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    combined_df = combined_df.sort_values(
        by=[
            "video_id",
            "tracking_timestamp",
        ]
    )

    return combined_df


# =====================================
# CALCULATE TRACKING FEATURES
# =====================================

def calculate_tracking_features(
    tracking_df,
):

    numeric_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "engagement_rate",
        "virality_score",
        "sentiment_score",
        "emotion_score",
        "toxicity_score",
        "pharma_risk_score",
    ]

    for column in numeric_columns:

        if column not in tracking_df.columns:

            tracking_df[column] = 0

        tracking_df[column] = (
            safe_numeric(
                tracking_df[column]
            )
        )

    # SORT
    tracking_df = (
        tracking_df.sort_values(
            by=[
                "video_id",
                "tracking_timestamp",
            ]
        )
    )

    # PREVIOUS VALUES
    tracking_df["prev_views"] = (
        tracking_df.groupby(
            "video_id"
        )["view_count"].shift(1)
    )

    tracking_df["prev_likes"] = (
        tracking_df.groupby(
            "video_id"
        )["like_count"].shift(1)
    )

    tracking_df["prev_comments"] = (
        tracking_df.groupby(
            "video_id"
        )["comment_count"].shift(1)
    )

    tracking_df["prev_engagement"] = (
        tracking_df.groupby(
            "video_id"
        )["engagement_rate"].shift(1)
    )

    tracking_df["prev_timestamp"] = (
        tracking_df.groupby(
            "video_id"
        )[
            "tracking_timestamp"
        ].shift(1)
    )

    # TIME DIFFERENCE
    tracking_df["time_diff_hours"] = (
        (
            tracking_df[
                "tracking_timestamp"
            ] -
            tracking_df[
                "prev_timestamp"
            ]
        )
        .dt.total_seconds()
        / 3600
    )

    tracking_df[
        "time_diff_hours"
    ] = tracking_df[
        "time_diff_hours"
    ].replace(
        0,
        np.nan,
    )

    # VELOCITY
    tracking_df["view_velocity"] = (
        (
            tracking_df[
                "view_count"
            ] -
            tracking_df[
                "prev_views"
            ]
        )
        /
        tracking_df[
            "time_diff_hours"
        ]
    )

    tracking_df["like_velocity"] = (
        (
            tracking_df[
                "like_count"
            ] -
            tracking_df[
                "prev_likes"
            ]
        )
        /
        tracking_df[
            "time_diff_hours"
        ]
    )

    tracking_df[
        "comment_velocity"
    ] = (
        (
            tracking_df[
                "comment_count"
            ] -
            tracking_df[
                "prev_comments"
            ]
        )
        /
        tracking_df[
            "time_diff_hours"
        ]
    )

    tracking_df[
        "engagement_velocity"
    ] = (
        (
            tracking_df[
                "engagement_rate"
            ] -
            tracking_df[
                "prev_engagement"
            ]
        )
        /
        tracking_df[
            "time_diff_hours"
        ]
    )

    # ACCELERATION
    tracking_df[
        "prev_view_velocity"
    ] = (
        tracking_df.groupby(
            "video_id"
        )[
            "view_velocity"
        ].shift(1)
    )

    tracking_df[
        "growth_acceleration"
    ] = (
        tracking_df[
            "view_velocity"
        ] -
        tracking_df[
            "prev_view_velocity"
        ]
    )

    # ENGAGEMENT MOMENTUM
    tracking_df[
        "engagement_momentum"
    ] = (
        (
            tracking_df[
                "like_velocity"
            ].fillna(0)
        )
        +
        (
            tracking_df[
                "comment_velocity"
            ].fillna(0)
        )
    )

    # TREND SCORE
    tracking_df["trend_score"] = (
        (
            tracking_df[
                "view_velocity"
            ].fillna(0)
            * 0.35
        )
        +
        (
            tracking_df[
                "like_velocity"
            ].fillna(0)
            * 0.15
        )
        +
        (
            tracking_df[
                "comment_velocity"
            ].fillna(0)
            * 0.15
        )
        +
        (
            tracking_df[
                "engagement_momentum"
            ].fillna(0)
            * 0.15
        )
        +
        (
            tracking_df[
                "virality_score"
            ].fillna(0)
            * 0.10
        )
        +
        (
            tracking_df[
                "pharma_risk_score"
            ].fillna(0)
            * 0.10
        )
    )

    # TREND LABEL
    threshold = (
        tracking_df[
            "trend_score"
        ]
        .fillna(0)
        .quantile(0.95)
    )

    tracking_df["trend_label"] = np.where(
        tracking_df[
            "trend_score"
        ] >= threshold,
        "trending",
        "normal",
    )

    # SPIKE DETECTION
    tracking_df["spike_detected"] = np.where(
        tracking_df[
            "growth_acceleration"
        ] > (
            tracking_df[
                "growth_acceleration"
            ]
            .fillna(0)
            .quantile(0.95)
        ),
        1,
        0,
    )

    # FILL NULLS
    fill_columns = [
        "view_velocity",
        "like_velocity",
        "comment_velocity",
        "engagement_velocity",
        "growth_acceleration",
        "engagement_momentum",
    ]

    for column in fill_columns:

        tracking_df[column] = (
            tracking_df[column]
            .fillna(0)
        )

    return tracking_df


# =====================================
# SAVE OUTPUTS
# =====================================

def save_outputs(
    tracking_history_df,
    tracking_features_df,
):

    # SAVE HISTORY CSV
    tracking_history_df.to_csv(
        TRACKING_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved tracking history:\n"
        f"{TRACKING_HISTORY_FILE}"
    )

    # SAVE FEATURES CSV
    tracking_features_df.to_csv(
        TRACKING_FEATURES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved tracking features:\n"
        f"{TRACKING_FEATURES_FILE}"
    )

    # SAVE PARQUET
    try:

        tracking_features_df.to_parquet(
            "youtube_tracking_features.parquet",
            index=False,
        )

        print(
            "\nSaved parquet tracking features."
        )

    except Exception as error:

        print(
            "\nParquet save failed:"
        )

        print(str(error))


# =====================================
# MAIN
# =====================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YouTube Tracking Updater"
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional feature CSV path"
        ),
    )

    args = parser.parse_args()

    # FIND FEATURE FILE
    if args.input:

        feature_file = Path(
            args.input
        )

    else:

        print(
            "\nSearching for latest "
            "*_features.csv..."
        )

        feature_file = (
            find_latest_feature_file()
        )

    print(
        f"\nUsing feature file:\n"
        f"{feature_file}"
    )

    # LOAD FEATURE DATA
    feature_df = load_feature_dataset(
        feature_file
    )

    print(
        f"\nLoaded rows: {len(feature_df)}"
    )

    # MERGE LIVE METRICS FROM SNAPSHOTS
    snapshots_df = load_latest_snapshots(
        Path(SNAPSHOTS_FILE)
    )

    feature_df = merge_live_metrics(
        feature_df,
        snapshots_df,
    )

    # BUILD SNAPSHOT
    snapshot_df = (
        build_tracking_snapshot(
            feature_df
        )
    )

    print(
        "\nCreated tracking snapshot."
    )

    # LOAD HISTORY
    history_df = load_tracking_history(
        Path(
            TRACKING_HISTORY_FILE
        )
    )

    # APPEND HISTORY
    tracking_history_df = (
        append_tracking_history(
            history_df,
            snapshot_df,
        )
    )

    print(
        "\nTracking history updated."
    )

    # CALCULATE TRACKING FEATURES
    tracking_features_df = (
        calculate_tracking_features(
            tracking_history_df
        )
    )

    print(
        "\nTracking features calculated."
    )

    # SAVE OUTPUTS
    save_outputs(
        tracking_history_df,
        tracking_features_df,
    )

    print(
        "\nTracking update completed "
        "successfully."
    )


if __name__ == "__main__":
    main()
