"""
YouTube Monitoring Pipeline Orchestrator

Runs the full pipeline on a configurable schedule, or once for a backfill.
Run this from pipeline_stable/ for production use.

Usage:
    python youtube_pipeline_runner.py                          # runs every 30 min
    python youtube_pipeline_runner.py --run-once               # single run, then exit
    python youtube_pipeline_runner.py --run-once --lookback-hours 24   # 24h backfill
    python youtube_pipeline_runner.py --interval-minutes 60

Pipeline stage order:
    1. youtube_data_extraction.py       --incremental          [API quota: ~5,000 units]
    2. youtube_tracking_extraction.py                          [API quota: cheap, ~1 unit/50 videos]
    3. youtube_data_cleaning.py         --incremental
    4. youtube_feature_engineering.py   --incremental
    5. youtube_tracking_updater.py      (merges live snapshots + NLP features → velocity)
    6. youtube_narrative_detection.py
    7. youtube_virality_predict.py      (requires youtube_virality_model.pkl)
    8. youtube_alert_engine.py

First-time setup:
    1. Create .env with YOUTUBE_API_KEY=your_key
    2. Run stages 1–8 manually in order (see CLAUDE.md for exact commands)
    3. Train the virality model:  python youtube_virality_model.py
    4. Start continuous monitoring:  python youtube_pipeline_runner.py
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple


LOG_FILE = "pipeline.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


def run_stage(script: str, extra_args: Optional[List[str]] = None) -> bool:
    cmd = [sys.executable, script] + (extra_args or [])
    logging.info("  >> %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error("Stage failed: %s (exit %d)", script, result.returncode)
        return False
    return True


def build_stages(published_after: str) -> List[Tuple[str, List[str]]]:
    stages: List[Tuple[str, List[str]]] = [
        # 1. Find new videos published since last run
        (
            "youtube_data_extraction.py",
            ["--published-after", published_after, "--incremental"],
        ),
        # 2. Snapshot current API metrics for all tracked videos
        (
            "youtube_tracking_extraction.py",
            ["--input", "youtube_video_data.csv", "--comment-pages", "0", "--skip-transcripts"],
        ),
        # 3. Clean and translate only new rows
        (
            "youtube_data_cleaning.py",
            ["--incremental"],
        ),
        # 4. NLP feature engineering for new rows only
        (
            "youtube_feature_engineering.py",
            ["--incremental"],
        ),
        # 5. Compute velocity / acceleration from tracking history
        (
            "youtube_tracking_updater.py",
            [],
        ),
        # 6. Narrative clustering and topic modelling
        (
            "youtube_narrative_detection.py",
            [],
        ),
        # 7. Virality prediction (inference — model must exist)
        (
            "youtube_virality_predict.py",
            [],
        ),
        # 8. Alert generation
        (
            "youtube_alert_engine.py",
            [],
        ),
    ]

    if not Path("youtube_virality_model.pkl").exists():
        logging.warning(
            "youtube_virality_model.pkl not found — skipping prediction and alerts.\n"
            "Train the model once data is collected:  python youtube_virality_model.py"
        )
        stages = [
            (s, a)
            for s, a in stages
            if s not in ("youtube_virality_predict.py", "youtube_alert_engine.py")
        ]

    return stages


def run_pipeline(lookback_hours: int) -> None:
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    logging.info("=" * 64)
    logging.info(
        "Pipeline started — lookback %dh — since %s",
        lookback_hours,
        published_after,
    )
    logging.info("=" * 64)

    for script, extra_args in build_stages(published_after):
        if not Path(script).exists():
            logging.warning("Script not found, skipping: %s", script)
            continue
        run_stage(script, extra_args)

    logging.info("Pipeline completed at %s.", datetime.now().strftime("%H:%M:%S"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube Monitoring Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Minutes between pipeline runs.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=2,
        help="Hours to look back when searching for new videos each run.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run the pipeline once and exit.",
    )
    args = parser.parse_args()

    setup_logging()

    if args.run_once:
        run_pipeline(args.lookback_hours)
        return

    logging.info(
        "Scheduler started — interval %d min, lookback %dh.",
        args.interval_minutes,
        args.lookback_hours,
    )

    while True:
        run_pipeline(args.lookback_hours)
        logging.info("Next run in %d minutes.", args.interval_minutes)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
