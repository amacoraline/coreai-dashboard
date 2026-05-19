import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        YouTubeTranscriptApi,
    )
except ImportError:
    NoTranscriptFound = None
    TranscriptsDisabled = None
    YouTubeTranscriptApi = None


API_BASE_URL = "https://www.googleapis.com/youtube/v3"

SNAPSHOT_COLUMNS = [
    "video_id",
    "keyword",
    "snapshot_time",
    "published_at",
    "minutes_since_publish",
    "title",
    "channel_id",
    "channel_title",
    "channel_subscriber_count",
    "url",
    "duration",
    "view_count",
    "like_count",
    "favorite_count",
    "comment_count",
    "share_count",
    "comments_text_full",
    "comments_count_fetched",
    "top_comment_like_total",
    "transcript_language",
    "transcript",
]


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def youtube_get(endpoint: str, params: Dict[str, object], api_key: str) -> Dict:
    query = urlencode({**params, "key": api_key})
    request = Request(f"{API_BASE_URL}/{endpoint}?{query}")

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"YouTube API request failed for {endpoint} with HTTP {exc.code}: {error_body}"
        ) from exc


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def append_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    if not rows:
        return

    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def fetch_video_details(video_ids: List[str], api_key: str, sleep_seconds: float) -> Dict[str, Dict]:
    videos = {}

    for batch in chunks(video_ids, 50):
        data = youtube_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails,status",
                "id": ",".join(batch),
                "maxResults": 50,
            },
            api_key,
        )

        for item in data.get("items", []):
            videos[item["id"]] = item

        time.sleep(sleep_seconds)

    return videos


def fetch_channel_details(channel_ids: List[str], api_key: str, sleep_seconds: float) -> Dict[str, Dict]:
    channels = {}

    for batch in chunks(channel_ids, 50):
        data = youtube_get(
            "channels",
            {
                "part": "snippet,statistics",
                "id": ",".join(batch),
                "maxResults": 50,
            },
            api_key,
        )

        for item in data.get("items", []):
            channels[item["id"]] = item

        time.sleep(sleep_seconds)

    return channels


def fetch_comments(video_id: str, api_key: str, max_pages: int, sleep_seconds: float) -> List[Dict[str, object]]:
    comments = []
    next_page_token = None

    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance",
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            data = youtube_get("commentThreads", params, api_key)
        except Exception as exc:
            print(f"Comments skipped for {video_id}: {exc}")
            break

        for item in data.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append(
                {
                    "text": snippet.get("textDisplay", ""),
                    "like_count": int(snippet.get("likeCount", 0) or 0),
                }
            )

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(sleep_seconds)

    return comments


def build_comments_summary(comments: List[Dict[str, object]]) -> Dict[str, object]:
    comment_texts = [str(comment.get("text", "")).strip() for comment in comments if str(comment.get("text", "")).strip()]
    return {
        "comments_text_full": "\n\n".join(comment_texts),
        "comments_count_fetched": len(comments),
        "top_comment_like_total": sum(int(comment.get("like_count", 0) or 0) for comment in comments),
    }


def fetch_english_transcript(video_id: str) -> Dict[str, str]:
    if YouTubeTranscriptApi is None:
        return {"transcript_language": "", "transcript": ""}

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                transcript = transcript_list.find_transcript(["en"])
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_manually_created_transcript(["en"])
                except NoTranscriptFound:
                    transcript = None

            if transcript is None:
                for candidate in transcript_list:
                    if getattr(candidate, "is_translatable", False):
                        transcript = candidate.translate("en")
                        break

            if transcript is None:
                return {"transcript_language": "", "transcript": ""}

            entries = transcript.fetch()
            full_text = " ".join(entry.text.strip() for entry in entries if entry.text.strip())
            return {
                "transcript_language": getattr(transcript, "language_code", "en"),
                "transcript": full_text,
            }

        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            full_text = " ".join(str(entry.get("text", "")).strip() for entry in entries if str(entry.get("text", "")).strip())
            return {
                "transcript_language": "en",
                "transcript": full_text,
            }

        return {"transcript_language": "", "transcript": ""}
    except Exception as exc:
        if TranscriptsDisabled and isinstance(exc, TranscriptsDisabled):
            return {"transcript_language": "", "transcript": ""}
        if NoTranscriptFound and isinstance(exc, NoTranscriptFound):
            return {"transcript_language": "", "transcript": ""}
        print(f"Transcript skipped for {video_id}: {exc}")
        return {"transcript_language": "", "transcript": ""}


def parse_minutes_since_publish(published_at: str, snapshot_time: datetime) -> Optional[float]:
    if not published_at:
        return None

    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return None

    return round((snapshot_time - published_dt).total_seconds() / 60, 2)


def load_tracking_targets(input_path: Path) -> List[Dict[str, str]]:
    rows = read_csv(input_path)
    if not rows:
        raise SystemExit(f"No tracking input rows found in {input_path}")

    seen = set()
    targets = []
    for row in rows:
        video_id = row.get("video_id", "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        targets.append(
            {
                "video_id": video_id,
                "keyword": row.get("keyword", "").strip(),
            }
        )
    return targets


def build_snapshot_row(
    keyword: str,
    item: Dict,
    channel_details: Optional[Dict],
    transcript_details: Dict[str, str],
    comments_summary: Dict[str, object],
    snapshot_time: datetime,
) -> Dict[str, object]:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    channel_statistics = (channel_details or {}).get("statistics", {})

    published_at = snippet.get("publishedAt", "")
    minutes_since_publish = parse_minutes_since_publish(published_at, snapshot_time)
    video_id = item.get("id", "")

    return {
        "video_id": video_id,
        "keyword": keyword,
        "snapshot_time": snapshot_time.isoformat(),
        "published_at": published_at,
        "minutes_since_publish": minutes_since_publish if minutes_since_publish is not None else "",
        "title": snippet.get("title", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "channel_subscriber_count": channel_statistics.get("subscriberCount", 0),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "duration": content_details.get("duration", ""),
        "view_count": statistics.get("viewCount", 0),
        "like_count": statistics.get("likeCount", 0),
        "favorite_count": statistics.get("favoriteCount", 0),
        "comment_count": statistics.get("commentCount", 0),
        "share_count": "not_available",
        "comments_text_full": comments_summary.get("comments_text_full", ""),
        "comments_count_fetched": comments_summary.get("comments_count_fetched", 0),
        "top_comment_like_total": comments_summary.get("top_comment_like_total", 0),
        "transcript_language": transcript_details.get("transcript_language", ""),
        "transcript": transcript_details.get("transcript", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track YouTube video metrics over time.")
    parser.add_argument("--input", default="youtube_video_data.csv", help="Input CSV containing at least video_id and keyword.")
    parser.add_argument("--output", default="youtube_video_snapshots.csv", help="Snapshot CSV output path.")
    parser.add_argument("--comment-pages", type=int, default=1, help="Comment pages to fetch per video snapshot.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--skip-transcripts", action="store_true", help="Disable transcript fetching.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("Set YOUTUBE_API_KEY in .env or PowerShell first.")

    targets = load_tracking_targets(Path(args.input))
    video_ids = [target["video_id"] for target in targets]
    keyword_lookup = {target["video_id"]: target["keyword"] for target in targets}
    snapshot_time = datetime.now(timezone.utc)

    details = fetch_video_details(video_ids, api_key, args.sleep)
    channel_ids = list(
        dict.fromkeys(
            item.get("snippet", {}).get("channelId", "")
            for item in details.values()
            if item.get("snippet", {}).get("channelId", "")
        )
    )
    channel_details = fetch_channel_details(channel_ids, api_key, args.sleep)

    snapshot_rows = []
    for video_id in video_ids:
        item = details.get(video_id)
        if not item:
            continue

        transcript_details = {"transcript_language": "", "transcript": ""}
        if not args.skip_transcripts:
            transcript_details = fetch_english_transcript(video_id)

        comments_summary = {
            "comments_text_full": "",
            "comments_count_fetched": 0,
            "top_comment_like_total": 0,
        }
        if args.comment_pages > 0:
            comments = fetch_comments(video_id, api_key, args.comment_pages, args.sleep)
            comments_summary = build_comments_summary(comments)

        channel_id = item.get("snippet", {}).get("channelId", "")
        snapshot_rows.append(
            build_snapshot_row(
                keyword=keyword_lookup.get(video_id, ""),
                item=item,
                channel_details=channel_details.get(channel_id),
                transcript_details=transcript_details,
                comments_summary=comments_summary,
                snapshot_time=snapshot_time,
            )
        )

    append_csv(Path(args.output), snapshot_rows, SNAPSHOT_COLUMNS)
    print(f"Saved {len(snapshot_rows)} snapshot rows to {args.output}")


if __name__ == "__main__":
    main()
