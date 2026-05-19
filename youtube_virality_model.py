
import argparse
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier


# =====================================
# CONFIG
# =====================================

MODEL_FILE = "youtube_virality_model.pkl"

PREDICTION_FILE = (
    "youtube_virality_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    "youtube_feature_importance.csv"
)


# =====================================
# UTILITY FUNCTIONS
# =====================================

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


def find_latest_narrative_file():

    current_dir = Path.cwd()

    files = list(
        current_dir.glob(
            "youtube_narrative_detection.csv"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No narrative detection file found."
        )

    latest_file = max(
        files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    return latest_file


# =====================================
# FEATURE SELECTION
# =====================================

def select_features(df):

    target_column = "viral_label"

    feature_columns = [

        # ENGAGEMENT
        "engagement_rate",
        "views_per_hour",
        "likes_per_hour",
        "comments_per_hour",
        "view_velocity",
        "like_velocity",
        "comment_velocity",
        "growth_acceleration",
        "trend_score",

        # CONTENT
        "text_length",
        "word_count",
        "emoji_count",
        "question_mark_count",
        "exclamation_count",

        # NLP
        "sentiment_score",
        "emotion_score",
        "toxicity_score",

        # RISK
        "pharma_risk_score",

        # NARRATIVE
        "narrative_similarity_score",
        "cluster_volume",
        "narrative_trend_score",

        # CATEGORICAL
        "sentiment_label",
        "emotion_label",
        "toxicity_label",
        "pharma_risk_label",
        "trend_label",
        "emerging_narrative",
    ]

    existing_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:

        print(
            "\nMissing features skipped:"
        )

        print(missing_features)

    if target_column not in df.columns:

        raise ValueError(
            "viral_label column missing."
        )

    X = df[
        existing_features
    ].copy()

    y = df[
        target_column
    ].astype(int)

    return (
        X,
        y,
        existing_features,
    )


# =====================================
# PREPROCESSING
# =====================================

def build_preprocessor(X):

    numeric_features = (
        X.select_dtypes(
            include=[
                np.number
            ]
        ).columns.tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude=[
                np.number
            ]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# =====================================
# TRAIN MODEL
# =====================================

def train_model(
    X_train,
    y_train,
    preprocessor,
):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print(
        "\nTraining virality model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


# =====================================
# EVALUATE MODEL
# =====================================

METRICS_FILE = "youtube_model_metrics.csv"


def evaluate_model(
    pipeline,
    X_test,
    y_test,
):

    print(
        "\nEvaluating model..."
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    y_arr = np.array(y_test)
    p_arr = np.array(probabilities)

    # ── Classification metrics ──
    accuracy  = accuracy_score(y_arr, predictions)
    roc_auc   = roc_auc_score(y_arr, p_arr)
    precision = precision_score(y_arr, predictions, zero_division=0)
    recall    = recall_score(y_arr, predictions, zero_division=0)
    f1        = f1_score(y_arr, predictions, zero_division=0)
    mcc       = matthews_corrcoef(y_arr, predictions)

    # ── Probabilistic metrics ──
    brier     = brier_score_loss(y_arr, p_arr)
    ll        = log_loss(y_arr, p_arr)

    # ── Regression-style metrics on probabilities vs labels ──
    rmse = np.sqrt(mean_squared_error(y_arr, p_arr))
    mae  = mean_absolute_error(y_arr, p_arr)

    # MAPE: only on viral samples (y=1) to avoid division-by-zero
    viral_mask = y_arr == 1
    if viral_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (y_arr[viral_mask] - p_arr[viral_mask])
                    / y_arr[viral_mask]
                )
            ) * 100
        )
    else:
        mape = float("nan")

    # ── Print ──
    sep = "-" * 42

    print(f"\n{sep}")
    print("  MODEL EVALUATION METRICS")
    print(sep)
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  MCC                : {mcc:.4f}")
    print(sep)
    print(f"  Brier Score        : {brier:.4f}  (lower = better)")
    print(f"  Log Loss           : {ll:.4f}  (lower = better)")
    print(sep)
    print(f"  RMSE (prob vs y)   : {rmse:.4f}")
    print(f"  MAE  (prob vs y)   : {mae:.4f}")
    print(f"  MAPE (viral only)  : {mape:.2f}%")
    print(sep)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_arr,
            predictions,
            target_names=["non-viral", "viral"],
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    cm = confusion_matrix(y_arr, predictions)
    print(cm)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # ── Save metrics CSV ──
    metrics_df = pd.DataFrame([{
        "accuracy":    round(accuracy,  4),
        "roc_auc":     round(roc_auc,   4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1_score":    round(f1,        4),
        "mcc":         round(mcc,       4),
        "brier_score": round(brier,     4),
        "log_loss":    round(ll,        4),
        "rmse":        round(rmse,      4),
        "mae":         round(mae,       4),
        "mape_viral":  round(mape, 2) if not np.isnan(mape) else None,
        "test_samples": len(y_arr),
        "viral_in_test": int(viral_mask.sum()),
    }])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved metrics: {METRICS_FILE}"
    )

    return probabilities


# =====================================
# FEATURE IMPORTANCE
# =====================================

def save_feature_importance(
    pipeline,
    feature_names,
):

    model = pipeline.named_steps[
        "model"
    ]

    importance_scores = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                importance_scores[
                    : len(feature_names)
                ]
            ),
        }
    )

    importance_df = (
        importance_df.sort_values(
            by="importance",
            ascending=False,
        )
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"\nSaved feature importance:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # PLOT
    top_features = (
        importance_df.head(15)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Virality Features"
    )

    plt.tight_layout()

    plt.savefig(
        "youtube_feature_importance.png"
    )

    print(
        "\nSaved feature importance plot."
    )


# =====================================
# SAVE PREDICTIONS
# =====================================

def save_predictions(
    df,
    probabilities,
):

    df["viral_probability"] = (
        probabilities
    )

    df["predicted_viral"] = np.where(
        df["viral_probability"] >= 0.7,
        1,
        0,
    )

    df.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved predictions:\n"
        f"{PREDICTION_FILE}"
    )


# =====================================
# MAIN
# =====================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YouTube Virality Model"
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional narrative CSV"
        ),
    )

    args = parser.parse_args()

    # FIND INPUT FILE
    if args.input:

        input_path = Path(
            args.input
        )

    else:

        print(
            "\nSearching for "
            "youtube_narrative_detection.csv..."
        )

        input_path = (
            find_latest_narrative_file()
        )

    print(
        f"\nUsing file:\n"
        f"{input_path}"
    )

    # LOAD DATA
    df = load_dataset(
        input_path
    )

    print(
        f"\nLoaded rows: {len(df)}"
    )

    # FEATURES
    (
        X,
        y,
        feature_names,
    ) = select_features(df)

    # SPLIT
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTrain rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    # PREPROCESSOR
    preprocessor = (
        build_preprocessor(X)
    )

    # TRAIN
    pipeline = train_model(
        X_train,
        y_train,
        preprocessor,
    )

    # EVALUATE
    probabilities = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    # FEATURE IMPORTANCE
    save_feature_importance(
        pipeline,
        feature_names,
    )

    # SAVE MODEL
    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    print(
        f"\nSaved model:\n"
        f"{MODEL_FILE}"
    )

    # SAVE PREDICTIONS
    save_predictions(
        df.iloc[
            X_test.index
        ].copy(),
        probabilities,
    )

    print(
        "\nVirality model training "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
