# CoreAI — YouTube Pharma Media Monitoring Platform

A pharma-focused YouTube intelligence platform that continuously ingests video and comment data from the YouTube Data API v3, runs multi-stage NLP analysis, clusters narratives, predicts virality using a trained XGBoost model, generates risk-adjusted alerts, and visualises everything in a real-time Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Environment Setup](#4-environment-setup)
5. [File-by-File Reference](#5-file-by-file-reference)
   - [youtube_data_extraction.py](#51-youtube_data_extractionpy)
   - [youtube_tracking_extraction.py](#52-youtube_tracking_extractionpy)
   - [youtube_data_cleaning.py](#53-youtube_data_cleaningpy)
   - [youtube_feature_engineering.py](#54-youtube_feature_engineeringpy)
   - [youtube_tracking_updater.py](#55-youtube_tracking_updaterpy)
   - [youtube_narrative_detection.py](#56-youtube_narrative_detectionpy)
   - [youtube_virality_model.py](#57-youtube_virality_modelpy)
   - [youtube_virality_predict.py](#58-youtube_virality_predictpy)
   - [youtube_alert_engine.py](#59-youtube_alert_enginepy)
   - [youtube_pipeline_runner.py](#510-youtube_pipeline_runnerpy)
   - [youtube_dashboard.py](#511-youtube_dashboardpy)
6. [CSV Files Reference](#6-csv-files-reference)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Model Performance](#8-model-performance)
9. [Dashboard Pages](#9-dashboard-pages)
10. [API Quota Guide](#10-api-quota-guide)
11. [Deployment](#11-deployment)

---

## 1. Project Overview

**Purpose:** Internal pharma media intelligence tool for monitoring YouTube content related to Organon brands, competitor drugs, key conditions (IVF, psoriasis, migraine, menopause), and pharma safety signals.

**What it does end-to-end:**
- Searches YouTube for 25 curated pharma keywords every 30 minutes
- Fetches video metadata, transcripts, and comments via the YouTube Data API v3
- Cleans and translates all text to English
- Runs NLP: sentiment analysis, emotion detection, toxicity scoring, pharma safety keyword scoring
- Computes engagement velocity, growth acceleration, and trend scores over time
- Clusters videos into narrative groups using sentence embeddings (KMeans) and extracts LDA topic keywords per cluster
- Trains an XGBoost binary classifier to predict viral videos (top 5th percentile by virality score)
- Generates RAVS (Risk-Adjusted Virality Score) combining viral probability, toxicity, trend, pharma risk, and influencer amplification
- Raises alerts for viral content, toxic content, pharma safety risks, emerging narratives, growth spikes, and influencer amplification
- Displays everything in a six-page Streamlit intelligence dashboard

---

## 2. Folder Structure

```
media monitoring project/
│
├── pipeline_stable/         ← PRODUCTION — run all jobs from here
├── pipeline_dev/            ← SANDBOX — edit and test here first
│
├── youtube_data_extraction.py          ← root copy (source of truth)
├── youtube_tracking_extraction.py
├── youtube_data_cleaning.py
├── youtube_feature_engineering.py
├── youtube_tracking_updater.py
├── youtube_narrative_detection.py
├── youtube_virality_model.py
├── youtube_virality_predict.py
├── youtube_alert_engine.py
├── youtube_pipeline_runner.py
├── youtube_dashboard.py
│
├── .env                     ← YOUTUBE_API_KEY goes here (never commit)
└── README.md
```

**Workflow rule:** Always edit in `pipeline_dev/` first, test it, then copy working changes to `pipeline_stable/` and the root. Run production jobs from `pipeline_stable/` only.

---

## 3. Architecture & Data Flow

```
YouTube Data API v3
        │
        ▼
youtube_data_extraction.py
        │  → youtube_video_data.csv          (new videos, append-only)
        │
        ▼
youtube_tracking_extraction.py
        │  → youtube_video_snapshots.csv     (live metric snapshots, append-only)
        │
        ▼
youtube_data_cleaning.py
        │  → youtube_video_data_cleaned.csv  (translated + normalised, append-only)
        │
        ▼
youtube_feature_engineering.py
        │  → youtube_video_data_cleaned_features.csv / .parquet
        │    (NLP scores: sentiment, emotion, toxicity, pharma risk, virality label)
        │
        ▼
youtube_tracking_updater.py
        │  reads: *_cleaned_features.csv  (static NLP scores)
        │       + youtube_video_snapshots.csv (live counts)
        │  → youtube_tracking_history.csv   (time-series of all snapshots)
        │  → youtube_tracking_features.csv  (+ velocity, acceleration, trend scores)
        │
        ▼
youtube_narrative_detection.py
        │  → youtube_narrative_detection.csv / .parquet
        │    (cluster IDs, topic keywords, cluster_topic_name, similarity, trend)
        │
        ▼
youtube_virality_predict.py          ← requires youtube_virality_model.pkl
        │  → youtube_virality_predictions.csv
        │
        ▼
youtube_alert_engine.py
        │  → youtube_alerts.csv / .parquet
        │
        ▼
youtube_dashboard.py  (Streamlit — reads all CSVs, serves live UI)


─── SEPARATE (run manually, not part of the loop) ───────────────

youtube_virality_model.py
        │  trains on youtube_narrative_detection.csv
        │  → youtube_virality_model.pkl
        │  → youtube_feature_importance.csv / .png
        │  → youtube_model_metrics.csv
```

---

## 4. Environment Setup

**Create a `.env` file in `pipeline_stable/`:**
```
YOUTUBE_API_KEY=your_google_api_key_here
```

**Install dependencies:**
```bash
pip install pandas numpy transformers detoxify sentence-transformers scikit-learn xgboost joblib matplotlib streamlit plotly deep-translator youtube-transcript-api pyarrow
```

**First-time data collection (run from `pipeline_stable/`):**
```bash
cd pipeline_stable

# Collect 30 days of historical videos
python youtube_data_extraction.py --published-after 2026-04-01T00:00:00Z --incremental

# Snapshot live metrics
python youtube_tracking_extraction.py --input youtube_video_data.csv --comment-pages 0 --skip-transcripts

# Clean and normalise (skip translation for speed on first run)
python youtube_data_cleaning.py --incremental --skip-translation

# NLP feature engineering (30–60s model load on first run)
python youtube_feature_engineering.py --incremental

# Build tracking history
python youtube_tracking_updater.py

# Narrative clustering
python youtube_narrative_detection.py

# Train virality model (only needed once; retrain periodically)
python youtube_virality_model.py

# Predict virality
python youtube_virality_predict.py

# Generate alerts
python youtube_alert_engine.py

# Launch dashboard (separate terminal)
streamlit run youtube_dashboard.py
```

---

## 5. File-by-File Reference

---

### 5.1 `youtube_data_extraction.py`

**Purpose:** Stage 1 of the pipeline. Searches YouTube for pharma-related videos using keyword queries and saves all video metadata, channel info, transcripts, and comments to CSV.

**Takes input from:** YouTube Data API v3 (live network calls). Reads `.env` for the API key.

**Produces:** `youtube_video_data.csv`

**Key design decisions:**
- Uses raw `urllib` (no third-party HTTP library) to call the YouTube API — avoids adding heavy dependencies.
- Each keyword search calls `search.list` (100 quota units each), then `videos.list` (1 unit per 50 videos) and `channels.list` to enrich results.
- Flushes results to disk after **every keyword** — so if quota runs out mid-run, already-fetched data is not lost.
- `--incremental` flag: loads existing `video_id`s from the output file and skips them — prevents duplicate rows across runs.
- Transcript fetching uses `youtube-transcript-api` with auto-fallback to translated transcripts if no English version exists.
- Comments are concatenated into a single `comments_text_full` field per video for downstream NLP.
- `DEFAULT_KEYWORDS` contains 25 curated pharma terms (Organon brands, high-familiarity conditions, key competitors). Can be overridden with `--keywords` or `--keyword-file`.

**Key columns produced:**
`keyword, video_id, title, description, channel_id, channel_title, channel_subscriber_count, published_at, url, duration, view_count, like_count, comment_count, transcript, transcript_status, comments_text_full, extracted_at`

**Quota cost:** ~5,000 units per full run (25 keywords × 2 pages × 100 units). Leaves ~5,000 units/day for tracking.

---

### 5.2 `youtube_tracking_extraction.py`

**Purpose:** Stage 2. Re-fetches live metrics (view count, like count, comment count, subscriber count) for all videos already known to the system. Creates a time-stamped snapshot row for every video on every run — building a time-series for velocity calculations.

**Takes input from:** `youtube_video_data.csv` (reads `video_id` list from it). YouTube Data API v3.

**Produces:** `youtube_video_snapshots.csv` (append-only — one row per video per run)

**Key design decisions:**
- Uses only `videos.list` (1 unit per 50 videos) — very cheap on quota.
- Records `minutes_since_publish` at snapshot time — useful for normalising velocity.
- `--skip-transcripts` (default in pipeline runner) skips re-fetching transcripts on every cycle — transcripts are static and were already fetched in Stage 1.
- `--comment-pages 0` (default in pipeline runner) disables comment fetching on tracking runs to save quota.
- Writes a new snapshot row each run — the updater (Stage 5) picks the most recent per video.

**Key columns produced:**
`video_id, keyword, snapshot_time, published_at, minutes_since_publish, view_count, like_count, comment_count, channel_subscriber_count`

---

### 5.3 `youtube_data_cleaning.py`

**Purpose:** Stage 3. Cleans and normalises raw video data. Translates all text fields (title, description, transcript, comments, channel title, keyword) to English using Google Translate. Consolidates all translated text into a single `analysis_text_en` field for downstream NLP.

**Takes input from:** `youtube_video_data.csv`

**Produces:** `youtube_video_data_cleaned.csv`

**Key design decisions:**
- Uses `deep-translator` (free Google Translate wrapper) with retry logic (3 attempts), exponential backoff, and a translation cache to avoid re-translating identical strings.
- Text is chunked into 4,000-character blocks before translation (Google Translate limit).
- `--skip-translation` flag normalises fields without translating — useful for English-only datasets or testing.
- `--incremental` flag skips `video_id`s already present in the output file.
- ISO 8601 duration strings (e.g. `PT4M32S`) are parsed into `duration_seconds` and `duration_minutes`.
- **`analysis_text_en`** is the key output field — concatenation of: `keyword_en + title_en + description_en + transcript_en + comments_text_full_en + channel_title_en`. Everything downstream uses this.
- For each source text column, an `_en` version is appended (e.g. `title` → `title_en`).

**Key columns added:**
`keyword_en, title_en, description_en, transcript_en, comments_text_full_en, channel_title_en, analysis_text_en, duration_seconds, duration_minutes, duration_text`

---

### 5.4 `youtube_feature_engineering.py`

**Purpose:** Stage 4. The heaviest NLP stage. Loads three transformer models at startup and runs them on every row's `analysis_text_clean`. Computes temporal features, engagement metrics, content features, NLP scores, pharma risk scores, and the `viral_label` target for model training.

**Takes input from:** `youtube_video_data_cleaned.csv`

**Produces:** `youtube_video_data_cleaned_features.csv` and `.parquet`

**NLP Models loaded at startup (30–60 seconds first run):**

| Task | Model |
|------|-------|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` — LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` — anger, disgust, fear, joy, neutral, sadness, surprise |
| Toxicity | `Detoxify("original")` — scores toxicity, severe_toxicity, obscene, threat, insult, identity_attack |

**Feature groups built:**

- **Temporal:** `hours_since_publish`, `days_since_publish`, `publish_hour`, `publish_weekday`, `publish_month`, `is_weekend`
- **Engagement:** `engagement_rate` (likes+comments / views), `views_per_hour`, `likes_per_hour`, `comments_per_hour`
- **Content:** `text_length`, `word_count`, `emoji_count`, `question_mark_count`, `exclamation_count`
- **NLP:** `sentiment_label`, `sentiment_score`, `emotion_label`, `emotion_score`, `toxicity_label`, `toxicity_score`
- **Pharma Risk:** `pharma_risk_score` (count of safety signal keywords matched), `pharma_risk_label` (high_risk / low_risk). Uses 50+ curated keywords: "death", "side effect", "lawsuit", "recall", "FDA warning", "nexplanon side", etc.
- **Virality target:** `virality_score` = weighted sum of engagement_rate (40%) + views_per_hour (30%) + comments_per_hour (20%) + sentiment_score (10%). `viral_label` = 1 if in top 5th percentile of virality_score, else 0.

**Key design decisions:**
- Text is first normalised (`analysis_text_clean`) by lowercasing, removing URLs, @mentions, hashtags, and extra whitespace.
- All NLP models receive at most 512 characters (transformer token limit).
- `--incremental` skips `video_id`s already in the output CSV.

---

### 5.5 `youtube_tracking_updater.py`

**Purpose:** Stage 5. Merges static NLP feature data with live snapshot metrics, builds a cumulative time-series (tracking history), and computes velocity, acceleration, momentum, and trend scores across snapshots.

**Takes input from:**
- `*_cleaned_features.csv` (NLP scores — static, computed once per video)
- `youtube_video_snapshots.csv` (live view/like/comment counts — one row per video per pipeline run)

**Produces:**
- `youtube_tracking_history.csv` — full time-series of all snapshots ever taken (append-only)
- `youtube_tracking_features.csv` — same data + velocity/acceleration columns (used by all downstream stages)

**Key features computed:**

| Column | Formula |
|--------|---------|
| `view_velocity` | (current_views − prev_views) / hours_between_snapshots |
| `like_velocity` | (current_likes − prev_likes) / hours_between_snapshots |
| `comment_velocity` | (current_comments − prev_comments) / hours_between_snapshots |
| `growth_acceleration` | view_velocity − prev_view_velocity |
| `engagement_momentum` | like_velocity + comment_velocity |
| `trend_score` | weighted combination of velocity (35%), like velocity (15%), comment velocity (15%), momentum (15%), virality score (10%), pharma risk (10%) |
| `trend_label` | "trending" if trend_score ≥ 95th percentile, else "normal" |
| `spike_detected` | 1 if growth_acceleration ≥ 95th percentile |

**Key design decisions:**
- Globs for `*_cleaned_features.csv` (not `*_features.csv`) to avoid matching its own output file `youtube_tracking_features.csv`.
- On first run a video has only one snapshot — velocity is NaN (filled to 0). Velocity becomes meaningful from the second snapshot onward.
- Live snapshot metrics overwrite stale extraction-time counts from the features file, ensuring views/likes are always up-to-date.

---

### 5.6 `youtube_narrative_detection.py`

**Purpose:** Stage 6. Groups videos into narrative clusters using sentence embeddings, extracts LDA topic keywords for each cluster, computes narrative similarity scores, identifies emerging narratives, and assigns a human-readable `cluster_topic_name` to each cluster.

**Takes input from:** `youtube_tracking_features.csv`

**Produces:** `youtube_narrative_detection.csv` and `.parquet`

**Model loaded at startup:** `paraphrase-MiniLM-L3-v2` via `sentence-transformers`

**Pipeline steps:**

1. **Embeddings** — encode `analysis_text_clean` for all rows using `SentenceTransformer`. Output: 384-dimensional normalised embeddings.

2. **KMeans Clustering** — `MiniBatchKMeans` with cluster count auto-scaled: `min(50, max(5, len(rows) // 1000))`. Assigns each row a `narrative_cluster_id`.

3. **LDA Topic Modeling** — `CountVectorizer` (max 2,000 features, English stop words) → `LatentDirichletAllocation` with `min(10, max(3, len(rows) // 20))` topics. Each topic is represented by its top 10 keywords. Assigns `dominant_topic_id` and `topic_keywords` (comma-separated top words).

4. **Cluster Topic Name** — For each KMeans cluster, finds the modal `dominant_topic_id` (most common LDA topic within the cluster), then maps it to its keyword string. Stored as `cluster_topic_name` — a single human-readable label per narrative cluster (e.g. `"psoriasis, dupixent, skin, treatment, patients"`). This is what the RAVS Engine dashboard page displays.

5. **Similarity Scores** — cosine similarity matrix on up to 1,000 sampled embeddings. Each row gets a `narrative_similarity_score` (max similarity to any other row).

6. **Narrative Trend Score** — weighted combination of cluster_volume (40%) + virality_score (30%) + trend_score (30%). Rows in top 5th percentile are flagged as `emerging_narrative = "emerging"`.

**Key columns produced:**
`narrative_cluster_id, dominant_topic_id, topic_keywords, cluster_topic_name, narrative_similarity_score, cluster_volume, narrative_trend_score, emerging_narrative`

---

### 5.7 `youtube_virality_model.py`

**Purpose:** Standalone training script (NOT part of the continuous pipeline loop). Trains an XGBoost binary classifier on narrative detection data to predict which videos will go viral. Run once initially, then retrain periodically as more data accumulates.

**Takes input from:** `youtube_narrative_detection.csv`

**Produces:**
- `youtube_virality_model.pkl` — trained sklearn Pipeline (preprocessor + XGBClassifier), loaded by `youtube_virality_predict.py`
- `youtube_feature_importance.csv` — feature importances ranked
- `youtube_feature_importance.png` — bar chart of top 15 features
- `youtube_model_metrics.csv` — full evaluation metrics table

**Model architecture:**

```
sklearn Pipeline:
  ├── ColumnTransformer (preprocessor)
  │     ├── Numeric: SimpleImputer(median) → StandardScaler
  │     └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown=ignore)
  └── XGBClassifier
        ├── n_estimators=200
        ├── max_depth=6
        ├── learning_rate=0.05
        ├── subsample=0.8
        └── colsample_bytree=0.8
```

**Features used (26 total):**
- Engagement velocity: `engagement_rate`, `views_per_hour`, `likes_per_hour`, `comments_per_hour`, `view_velocity`, `like_velocity`, `comment_velocity`, `growth_acceleration`, `trend_score`
- Content: `text_length`, `word_count`, `emoji_count`, `question_mark_count`, `exclamation_count`
- NLP scores: `sentiment_score`, `emotion_score`, `toxicity_score`
- Pharma risk: `pharma_risk_score`
- Narrative: `narrative_similarity_score`, `cluster_volume`, `narrative_trend_score`
- Categorical: `sentiment_label`, `emotion_label`, `toxicity_label`, `pharma_risk_label`, `trend_label`, `emerging_narrative`

**Target:** `viral_label` (1 = top 5th percentile by virality_score, defined in feature engineering)

**Train/test split:** 80/20, stratified on `viral_label`, `random_state=42`

**Evaluation metrics printed and saved:**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| ROC-AUC | Ranking quality (0.5 = random, 1.0 = perfect) |
| Precision | Of predicted viral, how many truly were |
| Recall | Of actual viral videos, how many were caught |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient — best for imbalanced classes |
| Brier Score | Calibration of output probabilities (lower = better) |
| Log Loss | Penalises confident wrong predictions (lower = better) |
| RMSE | Root mean squared error between probabilities and true 0/1 labels |
| MAE | Mean absolute error between probabilities and true 0/1 labels |
| MAPE (viral only) | Mean absolute percentage error on viral-class rows only (avoids division by zero on non-viral rows) |

**Achieved metrics (on 11,882 training rows, 2,376 test rows):**
```
Accuracy:     0.9878
ROC-AUC:      0.9977
Precision:    0.9500
Recall:       0.7983
F1 Score:     0.8676
MCC:          0.8648
Brier Score:  0.0081
RMSE:         0.0898
MAPE viral:   19.03%
```

---

### 5.8 `youtube_virality_predict.py`

**Purpose:** Stage 7 of the pipeline. Inference-only — loads the saved model and predicts viral probability for all rows in the latest narrative detection file. Does NOT retrain.

**Takes input from:**
- `youtube_narrative_detection.csv`
- `youtube_virality_model.pkl`

**Produces:** `youtube_virality_predictions.csv`

**Key columns added:**
- `viral_probability` — continuous score 0.0–1.0 from `predict_proba`
- `predicted_viral` — binary flag: 1 if `viral_probability >= 0.7`, else 0

**Key design decisions:**
- Uses exactly the same 26 feature columns as the training script. Any missing columns are filled with 0 / "unknown" rather than crashing — makes the script robust to schema drift.
- The 0.7 threshold for `predicted_viral` is a deliberate choice: higher precision at the cost of some recall. Lower it to 0.5–0.55 in the code to catch more viral videos at the cost of more false positives.
- Preserves all original columns from the input and appends the two new prediction columns.

---

### 5.9 `youtube_alert_engine.py`

**Purpose:** Stage 8 and final pipeline stage. Applies rule-based alert detection across five risk dimensions, computes the RAVS score for every row, deduplicates, and outputs the top 100 alerts.

**Takes input from:** `youtube_virality_predictions.csv`

**Produces:** `youtube_alerts.csv` and `.parquet`

**Alert types:**

| Alert Type | Trigger | Priority |
|------------|---------|----------|
| `viral_emerging` | `viral_probability >= 0.80` | High |
| `toxicity_spike` | `toxicity_score >= 0.75` | High |
| `pharma_risk` | `pharma_risk_score >= 1` | Critical |
| `emerging_narrative` | `narrative_trend_score >= 95th percentile` | Medium |
| `growth_spike` | `growth_acceleration >= 95th percentile` | Medium |
| `influencer_amplification` | `subscriber_count >= 100,000` AND `trend_score >= 90th percentile` | High |

**RAVS Score (Risk-Adjusted Virality Score):**

```
RAVS = (viral_probability × 0.30)
     + (toxicity_score × 0.20)
     + (normalised trend_score × 0.20)
     + (normalised pharma_risk_score × 0.20)
     + (influencer_flag × 0.10)
```

Stored as `ravs_score` (0.0–1.0). Five sub-component scores are also computed on a 0–100 scale for dashboard display: `ravs_speed_of_engagement`, `ravs_regulatory_relevance`, `ravs_safety_implications`, `ravs_influencer_reach`, `ravs_topic_sensitivity`.

**Deduplication:** Rows are sorted by `alert_score` descending, then deduplicated on `(video_id, alert_type)` — one alert per video per alert type.

**Output columns (where available):**
`video_id, title, channel_title, alert_type, alert_priority, alert_reason, alert_score, ravs_score, ravs_speed_of_engagement, ravs_regulatory_relevance, ravs_safety_implications, ravs_influencer_reach, ravs_topic_sensitivity, viral_probability, trend_score, growth_acceleration, sentiment_label, emotion_label, toxicity_label, toxicity_score, pharma_risk_score, narrative_cluster_id, dominant_topic_id, topic_keywords, cluster_topic_name, cluster_volume, narrative_trend_score, view_count, like_count, comment_count, subscriber_count, tracking_timestamp`

---

### 5.10 `youtube_pipeline_runner.py`

**Purpose:** Orchestrator. Runs all 8 pipeline stages in sequence on a configurable schedule. The only file you need to start for continuous monitoring.

**Takes input from:** Nothing directly — calls each stage script as a subprocess.

**Produces:** `pipeline.log` (timestamped log of every stage run and exit code)

**Scheduling logic:**
- Runs all 8 stages in order
- Sleeps for `--interval-minutes` (default 30) between runs
- Each run computes `published_after` = now − `--lookback-hours` (default 2) so only recent videos are queried
- If `youtube_virality_model.pkl` is missing, skips stages 7 and 8 automatically and logs a warning

**CLI options:**
```bash
python youtube_pipeline_runner.py                           # every 30 min
python youtube_pipeline_runner.py --interval-minutes 60     # every 60 min
python youtube_pipeline_runner.py --run-once                # single run and exit
python youtube_pipeline_runner.py --run-once --lookback-hours 24  # 24h backfill
```

**Stage failure handling:** If any stage returns a non-zero exit code, it logs an error and continues to the next stage — a single failure does not abort the entire pipeline run.

---

### 5.11 `youtube_dashboard.py`

**Purpose:** Streamlit web application. Reads all output CSVs and renders a six-page intelligence dashboard. No pipeline logic runs inside the dashboard — it is purely a visualisation layer.

**Takes input from:**
- `youtube_narrative_detection.csv`
- `youtube_virality_predictions.csv`
- `youtube_alerts.csv`
- `youtube_video_data_cleaned_features.csv`

**Data cache:** `@st.cache_data(ttl=300)` — all CSVs are cached for 5 minutes. A "Refresh Data" button in the sidebar forces an immediate reload.

**Sidebar:** Navigation radio + Indication Area dropdown (11 pharma therapy areas). Selecting an indication filters all page data to videos matching its keyword list using regex against the `keyword` column.

**Pages:**

#### Dashboard (Overview)
Five KPI cards: Total Videos, Active Alerts, Emerging Narratives, High Risk (RAVS≥70), Avg Viral Score. Viral probability histogram. Sentiment distribution pie. Active alerts table.

#### Narrative Monitor
Cluster viral probability bar chart (top 15 clusters by avg viral probability). Topic volume vs virality scatter plot (bubble size = video count). Narrative feed table sorted by viral probability showing topic keywords, sentiment, emerging status.

#### Alerts
Critical / High / Medium alert counts. Full alert details table.

#### Virality Predictor
Three scenario radio buttons. Each surfaces a different video:
- **Scenario A (Off-label Claims)** → highest `pharma_risk_score`
- **Scenario B (Clinical Trial)** → highest `viral_probability`
- **Scenario C (Brand Sentiment)** → highest `toxicity_score`

Shows: Viral Probability KPI, Time to Virality estimate, RAVS Score, Response Window. Predicted engagement trajectory chart (logistic growth model + acceleration). Signal contribution bar chart. Event timing timeline.

#### RAVS Engine
RAVS component bars (Speed of Engagement, Regulatory Relevance, Safety Implications, Influencer Reach, Topic Sensitivity). RAVS gauge chart. Risk classification tier cards (Critical/Region/Low). **Top Scored Narratives table** — aggregates by `cluster_topic_name` showing average RAVS score and trend direction per narrative cluster (one row per narrative, not per video).

#### Pharmacovigilance
Avg Toxicity / Avg Pharma Risk / Avg Viral Score KPIs. Toxicity vs virality scatter. Pharma risk distribution bar chart.

---

## 6. CSV Files Reference

| File | Created by | Read by | Description |
|------|-----------|---------|-------------|
| `youtube_video_data.csv` | data_extraction | tracking_extraction, data_cleaning | Raw video metadata, transcripts, comments |
| `youtube_comment_data.csv` | data_extraction | data_cleaning | Raw comments (optional) |
| `youtube_video_snapshots.csv` | tracking_extraction | tracking_updater | Live metric snapshots, one row per video per run |
| `youtube_video_data_cleaned.csv` | data_cleaning | feature_engineering | Cleaned + translated text |
| `youtube_video_data_cleaned_features.csv` | feature_engineering | tracking_updater | + NLP scores, virality label |
| `youtube_tracking_history.csv` | tracking_updater | tracking_updater (next run) | Full time-series, all snapshots |
| `youtube_tracking_features.csv` | tracking_updater | narrative_detection | + velocity, acceleration, trend scores |
| `youtube_narrative_detection.csv` | narrative_detection | virality_model, virality_predict, dashboard | + cluster IDs, topic keywords, cluster_topic_name |
| `youtube_virality_predictions.csv` | virality_predict | alert_engine, dashboard | + viral_probability, predicted_viral |
| `youtube_alerts.csv` | alert_engine | dashboard | Top 100 risk alerts with RAVS scores |
| `youtube_virality_model.pkl` | virality_model | virality_predict | Trained XGBoost pipeline |
| `youtube_model_metrics.csv` | virality_model | — | Accuracy, ROC-AUC, F1, RMSE, MAPE etc. |
| `youtube_feature_importance.csv` | virality_model | — | Feature importances ranked |
| `pipeline.log` | pipeline_runner | — | Timestamped run log |

---

## 7. Running the Pipeline

**Continuous monitoring (recommended):**
```bash
cd pipeline_stable
python youtube_pipeline_runner.py                        # every 30 min
python youtube_pipeline_runner.py --interval-minutes 60  # every 60 min
```

**Single run:**
```bash
python youtube_pipeline_runner.py --run-once
```

**Manual stage-by-stage:**
```bash
python youtube_data_extraction.py --published-after 2026-05-13T00:00:00Z --incremental
python youtube_tracking_extraction.py --input youtube_video_data.csv --comment-pages 0 --skip-transcripts
python youtube_data_cleaning.py --incremental --skip-translation
python youtube_feature_engineering.py --incremental
python youtube_tracking_updater.py
python youtube_narrative_detection.py
python youtube_virality_predict.py
python youtube_alert_engine.py
streamlit run youtube_dashboard.py
```

**Retrain the virality model:**
```bash
python youtube_virality_model.py
```

---

## 8. Model Performance

Trained on 11,882 rows, tested on 2,376 rows (119 viral in test set — 5% class imbalance):

```
Accuracy:      0.9878   Overall correct predictions
ROC-AUC:       0.9977   Near-perfect viral/non-viral ranking
Precision:     0.9500   Of videos flagged viral, 95% truly were
Recall:        0.7983   Catches ~80% of actual viral videos
F1 Score:      0.8676   Balanced precision/recall
MCC:           0.8648   Strong score on imbalanced data
Brier Score:   0.0081   Very well calibrated probabilities
Log Loss:      0.0252   Model is confident and correct
RMSE:          0.0898   Probabilities close to true labels
MAPE viral:    19.03%   Average undershoot on viral-class rows
```

**Interpretation:** The model is production-ready. The recall of 0.80 means ~20% of viral events are missed. Lower the `viral_probability >= 0.7` threshold in `youtube_virality_predict.py` to 0.5–0.55 to catch more at the cost of some precision.

---

## 9. Dashboard Pages

| Page | Primary Use Case |
|------|-----------------|
| Dashboard | Executive overview — KPIs, distribution charts, live alert table |
| Narrative Monitor | Cluster analysis — which narrative groups are most viral, topic volume scatter |
| Alerts | Operational — full alert feed for comms/regulatory teams |
| Virality Predictor | Scenario modelling — engagement trajectory, response window, signal contribution |
| RAVS Engine | Risk prioritisation — component breakdown, tier classification, top narrative risks |
| Pharmacovigilance | Safety signal monitoring — toxicity vs virality, pharma risk distribution |

---

## 10. API Quota Guide

Default quota: **10,000 units/day** per Google Cloud project.

| Operation | Cost |
|-----------|------|
| `search.list` (1 page, 50 results) | 100 units |
| `videos.list` (up to 50 video IDs) | 1 unit |
| `channels.list` (up to 50 channel IDs) | 1 unit |
| `commentThreads.list` (1 page) | 1 unit |

**Typical run cost:**
- Stage 1 (data extraction): 25 keywords × 2 pages = **~5,000 units**
- Stage 2 (tracking): 1 unit per 50 videos — very cheap
- **Remaining budget:** ~5,000 units/day for tracking

**Rules:**
- Do not exceed 50 keywords with `--max-pages 2` — daily quota will exhaust in one run
- Stages 3–8 require no API access — they run on local data
- If quota exhausts mid-run, Stage 1 saves what it fetched and exits with code 1; the pipeline runner logs the error and continues with stages 2–8 on existing data

---

## 11. Deployment

### Local (current setup)
Run `youtube_pipeline_runner.py` in a terminal. Dashboard available at `http://localhost:8501`.

### AWS EC2 Free Tier (dashboard only, accessible to others)
Deploy only the dashboard + CSV files on EC2 t2.micro (free for 12 months). Pipeline continues running on your local machine; upload updated CSVs to EC2 after each run.

**Required files on EC2:**
```
youtube_dashboard.py
youtube_alert_engine.py       (imported by dashboard)
requirements.txt
*.csv                         (latest pipeline output files)
```

**Minimal `requirements.txt`:**
```
pandas
numpy
streamlit
plotly
scikit-learn
xgboost
joblib
pyarrow
```

See deployment section in the conversation history for full step-by-step EC2 instructions including systemd service setup and SCP upload commands.

### Streamlit Community Cloud (free, public URL, no server)
Push dashboard + CSVs to a GitHub repo. Connect at share.streamlit.io. Zero server management. Data is static (no live pipeline on the server) — update by pushing new CSVs to GitHub.
