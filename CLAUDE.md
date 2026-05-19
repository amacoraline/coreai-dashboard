# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pharma-focused YouTube media monitoring platform. It continuously ingests video and comment data from the YouTube Data API v3, runs NLP analysis, clusters narratives, predicts virality, generates alerts, and visualises results in a Streamlit dashboard. The pipeline runs on a schedule and processes only new data each cycle (incremental mode).

## Environment Setup

Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_key_here
```

Python dependencies:
```
pip install pandas numpy transformers detoxify sentence-transformers scikit-learn xgboost joblib matplotlib streamlit plotly deep-translator youtube-transcript-api pyarrow
```

## Folder Structure

```
media monitoring project/
├── pipeline_stable/    ← production copy — run all jobs from here
├── pipeline_dev/       ← sandbox — safe to edit and experiment
└── (root)             ← source of truth for scripts; sync changes to both folders
```

Always run production jobs from `pipeline_stable/`. Edit in `pipeline_dev/` first to test, then copy working changes to `pipeline_stable/` and root.

## First-Time Setup

```bash
cd pipeline_stable

# Step 1: Collect initial data (backfill last 30 days)
python youtube_data_extraction.py --published-after 2026-04-01T00:00:00Z --incremental

# Step 2: Snapshot live metrics
python youtube_tracking_extraction.py --input youtube_video_data.csv --comment-pages 0 --skip-transcripts

# Step 3: Clean and translate
python youtube_data_cleaning.py --incremental --skip-translation

# Step 4: NLP feature engineering (30–60s model load on first run)
python youtube_feature_engineering.py --incremental

# Step 5: Build tracking history
python youtube_tracking_updater.py

# Step 6: Narrative clustering
python youtube_narrative_detection.py

# Step 7: Train the virality model (only needed once, retrain periodically)
python youtube_virality_model.py

# Step 8: Predict virality
python youtube_virality_predict.py

# Step 9: Generate alerts
python youtube_alert_engine.py

# Step 10: Launch dashboard (separate terminal)
streamlit run youtube_dashboard.py
```

## Running the Pipeline

### Continuous monitoring (recommended)
```bash
cd pipeline_stable
python youtube_pipeline_runner.py                        # every 30 min
python youtube_pipeline_runner.py --interval-minutes 60  # every 60 min
python youtube_pipeline_runner.py --run-once             # single run and exit
```

### Running individual stages manually (in order)
```bash
# Step 1 — finds NEW videos published since date (uses API quota)
python youtube_data_extraction.py --published-after 2026-05-13T00:00:00Z --incremental

# Step 2 — refreshes live view/like/comment counts for all known videos (uses API quota)
python youtube_tracking_extraction.py --input youtube_video_data.csv --comment-pages 0 --skip-transcripts

# Step 3 — cleans and translates text (no API needed)
python youtube_data_cleaning.py --incremental --skip-translation

# Step 4 — NLP sentiment / toxicity / virality features (no API needed)
python youtube_feature_engineering.py --incremental

# Step 5 — merges live snapshots + NLP features, computes velocity (no API needed)
python youtube_tracking_updater.py

# Step 6 — narrative clustering and topic modelling (no API needed)
python youtube_narrative_detection.py

# Step 7 — virality inference (requires trained model)
python youtube_virality_predict.py

# Step 8 — alert generation
python youtube_alert_engine.py

# Dashboard
streamlit run youtube_dashboard.py
```

### Train / retrain the virality model
```bash
python youtube_virality_model.py            # trains on youtube_narrative_detection.csv
```

### Snapshot live metrics for tracked videos (runs inside pipeline_runner automatically)
```bash
python youtube_tracking_extraction.py --input youtube_video_data.csv --comment-pages 0 --skip-transcripts
```

## Architecture and Data Flow

```
youtube_pipeline_runner.py  (orchestrator — runs on schedule)
  │
  ├─ youtube_data_extraction.py       --incremental          [uses API quota]
  │    → youtube_video_data.csv  (append-only, new videos only)
  │
  ├─ youtube_tracking_extraction.py                          [uses API quota]
  │    → youtube_video_snapshots.csv  (append-only, one row per video per run)
  │
  ├─ youtube_data_cleaning.py         --incremental
  │    → youtube_video_data_cleaned.csv  (append-only)
  │
  ├─ youtube_feature_engineering.py   --incremental
  │    → youtube_video_data_cleaned_features.csv / .parquet
  │
  ├─ youtube_tracking_updater.py
  │    reads: youtube_video_data_cleaned_features.csv  (NLP scores — static)
  │         + youtube_video_snapshots.csv              (live view/like/comment counts)
  │    → youtube_tracking_history.csv   (time-series of all snapshots)
  │    → youtube_tracking_features.csv  (+ velocity / acceleration columns)
  │
  ├─ youtube_narrative_detection.py
  │    → youtube_narrative_detection.csv / .parquet
  │
  ├─ youtube_virality_predict.py      (inference — loads youtube_virality_model.pkl)
  │    → youtube_virality_predictions.csv
  │
  └─ youtube_alert_engine.py
       → youtube_alerts.csv / .parquet
```

**Separate training script** (run manually, not part of the pipeline loop):
```
youtube_virality_model.py  →  youtube_virality_model.pkl
```

## Key Design Patterns

**Incremental processing**: `youtube_data_extraction.py`, `youtube_data_cleaning.py`, and `youtube_feature_engineering.py` all support `--incremental`. They load existing output to find already-processed `video_id`s and skip them. New rows are appended to the output CSV rather than overwriting it.

**Pipeline auto-detection**: When `--input` is omitted, each script globs for its expected input file by modification time. `youtube_tracking_updater.py` globs `*_cleaned_features.csv` (not `*_features.csv`) to avoid matching its own output `youtube_tracking_features.csv`. Follow this pattern when adding new stages.

**Train once, predict always**: `youtube_virality_model.py` trains and saves `youtube_virality_model.pkl`. The continuous pipeline calls `youtube_virality_predict.py` (inference only). Retrain the model manually after significant new data accumulates.

**RAVS (Risk-Adjusted Virality Score)**: Computed in `youtube_alert_engine.py:compute_ravs_score()` as a weighted combination of viral probability (30%), toxicity (20%), engagement trend (20%), pharma risk keyword density (20%), and influencer amplification (10% — channels with >100k subscribers).

**Virality label**: Defined in `youtube_feature_engineering.py` as the top 5th percentile by `virality_score`, which is a weighted sum of engagement rate, views/hour, comments/hour, and sentiment score.

**Alert thresholds**: Hardcoded constants at the top of `youtube_alert_engine.py` (`VIRALITY_THRESHOLD=0.80`, `TOXICITY_THRESHOLD=0.75`, `PHARMA_RISK_THRESHOLD=1`). Tune these to adjust sensitivity.

**Dashboard auto-refresh**: `youtube_dashboard.py` uses `@st.cache_data(ttl=300)` so data expires every 5 minutes. A "Refresh Data" button in the sidebar forces an immediate reload.

**CSV encoding**: All CSVs are written with `utf-8-sig` (BOM) for Excel compatibility.

**Dual output format**: Feature-rich stages save both `.csv` and `.parquet`. Parquet failures are caught and printed but do not abort the run.

**Text consolidation into `analysis_text_en`**: The cleaning step merges keyword, title, description, transcript, comments, and channel title (all translated to English) into a single `analysis_text_en` field. Downstream NLP stages operate on `analysis_text_clean` (normalised version).

**Narrative clustering**: Cluster count auto-scales between 5 and 50 based on `len(embeddings) // 1000`. LDA topic count is similarly bounded by corpus size.

## NLP Models Used

| Task | Model |
|------|-------|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` (LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive) |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` |
| Toxicity | `Detoxify("original")` |
| Embeddings / narrative clustering | `paraphrase-MiniLM-L3-v2` via `sentence-transformers` |

Models load at module import time in `youtube_feature_engineering.py` and `youtube_narrative_detection.py` — expect 30–60s startup delay on first run or after a cold cache.

## Default Pharma Keywords

`youtube_data_extraction.py` ships with **25 curated keywords** in `DEFAULT_KEYWORDS`, focused on Organon brands (nuvaring, nexplanon, mirena, propecia, singulair, renflexis, dupixent, humira…), high-familiarity conditions (IVF, psoriasis, migraine, menopause, hair loss…), and key competitors (abbvie, sanofi, regeneron, bayer, eli lilly). Override with:
```bash
--keywords "term1,term2"
--keyword-file keywords.txt   # one keyword per line, # for comments
```

**Do not exceed 50 keywords** with `--max-pages 2` or the daily quota will be exhausted in a single run.

## YouTube API Quota

Default quota is 10,000 units/day. `search.list` costs 100 units per call; `videos.list` costs 1 unit per 50 videos. With 25 keywords × 2 pages = **5,000 units per extraction run** — leaves 5,000 units for `youtube_tracking_extraction.py` (which uses only `videos.list` at 1 unit/50 videos and is very cheap).

- Each API key is tied to one Google Cloud **project** — keys from the same project share one quota pool.
- If quota is exhausted, Steps 3–9 still run fine (no API needed).
- Request a quota increase at Google Cloud Console → APIs & Services → YouTube Data API v3 → Quotas.
