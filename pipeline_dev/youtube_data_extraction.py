import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError
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

# Pharma-focused default keywords. Add or remove terms based on your monitoring brief.
DEFAULT_KEYWORDS = [
    # ── Organon company ────────────────────────────────────────
    "organon",

    # ── High-familiarity Organon brands ───────────────────────
    "nuvaring",           # widely searched contraceptive ring
    "nexplanon",          # top-searched contraceptive implant
    "mirena",             # leading hormonal IUD
    "propecia",           # well-known hair loss brand
    "finasteride",        # generic; high search volume
    "singulair",          # mass-market asthma/allergy brand
    "nasonex",            # common nasal spray brand
    "renflexis",          # biosimilar; growing awareness
    "dupixent",           # blockbuster competitor; benchmark
    "humira",             # highest-volume biologics term

    # ── High-familiarity conditions ───────────────────────────
    "IVF",                # very high public search volume
    "fertility",          # broad; catches influencer content
    "contraception",      # broad; policy + personal content
    "psoriasis",          # very high patient community volume
    "atopic dermatitis",  # clinical term; pairs with dupixent
    "migraine",           # extremely high search volume
    "menopause",          # large + growing audience
    "hair loss",          # high consumer search volume
    "biosimilar",         # policy + HCP conversations

    # ── Key competitor companies ──────────────────────────────
    "abbvie",             # humira/skyrizi parent; key benchmark
    "sanofi",             # dupixent parent
    "regeneron",          # dupixent partner
    "bayer",              # contraception / women's health overlap
    "eli lilly",          # migraine (emgality) / dermatology
]


VIDEO_COLUMNS = [
    "keyword",
    "video_id",
    "title",
    "description",
    "channel_id",
    "channel_title",
    "channel_subscriber_count",
    "published_at",
    "url",
    "duration",
    "category_id",
    "tags",
    "view_count",
    "like_count",
    "favorite_count",
    "comment_count",
    "definition",
    "caption",
    "licensed_content",
    "transcript_status",
    "transcript_language",
    "transcript",
    "share_count",
    "comments_text_full",
    "extracted_at",
]

COMMENT_COLUMNS = [
    "keyword",
    "video_id",
    "comment_id",
    "author",
    "author_channel_url",
    "text",
    "like_count",
    "published_at",
    "updated_at",
    "extracted_at",
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


def search_video_ids(
    keyword: str,
    api_key: str,
    max_pages: int,
    published_after: Optional[str],
    published_before: Optional[str],
    sleep_seconds: float,
) -> List[str]:
    video_ids = []
    next_page_token = None

    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": 50,
            "order": "date",
            "relevanceLanguage": "en",
            "safeSearch": "none",
        }

        if next_page_token:
            params["pageToken"] = next_page_token
        if published_after:
            params["publishedAfter"] = published_after
        if published_before:
            params["publishedBefore"] = published_before

        data = youtube_get("search", params, api_key)

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(sleep_seconds)

    return video_ids


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


def fetch_comments(
    keyword: str,
    video_id: str,
    api_key: str,
    max_pages: int,
    sleep_seconds: float,
) -> List[Dict[str, object]]:
    comments = []
    next_page_token = None
    extracted_at = datetime.now(timezone.utc).isoformat()

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
                    "keyword": keyword,
                    "video_id": video_id,
                    "comment_id": item.get("id", ""),
                    "author": snippet.get("authorDisplayName", ""),
                    "author_channel_url": snippet.get("authorChannelUrl", ""),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                    "updated_at": snippet.get("updatedAt", ""),
                    "extracted_at": extracted_at,
                }
            )

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(sleep_seconds)

    return comments


def build_comments_text_full(comments: List[Dict[str, object]]) -> str:
    return "\n\n".join(
        str(comment.get("text", "")).strip()
        for comment in comments
        if str(comment.get("text", "")).strip()
    )


def fetch_english_transcript(video_id: str) -> Dict[str, str]:
    if YouTubeTranscriptApi is None:
        return {
            "transcript_status": "transcript_library_missing",
            "transcript_language": "",
            "transcript": "",
        }

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
                return {
                    "transcript_status": "transcript_not_available",
                    "transcript_language": "",
                    "transcript": "",
                }

            entries = transcript.fetch()
            full_text = " ".join(entry.text.strip() for entry in entries if entry.text.strip())
            return {
                "transcript_status": "transcript_fetched",
                "transcript_language": getattr(transcript, "language_code", "en"),
                "transcript": full_text,
            }

        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            full_text = " ".join(str(entry.get("text", "")).strip() for entry in entries if str(entry.get("text", "")).strip())
            return {
                "transcript_status": "transcript_fetched",
                "transcript_language": "en",
                "transcript": full_text,
            }

        return {
            "transcript_status": "transcript_api_unsupported",
            "transcript_language": "",
            "transcript": "",
        }
    except Exception as exc:
        if TranscriptsDisabled and isinstance(exc, TranscriptsDisabled):
            return {
                "transcript_status": "transcript_disabled",
                "transcript_language": "",
                "transcript": "",
            }
        if NoTranscriptFound and isinstance(exc, NoTranscriptFound):
            return {
                "transcript_status": "transcript_not_found",
                "transcript_language": "",
                "transcript": "",
            }
        print(f"Transcript skipped for {video_id}: {exc}")
        return {
            "transcript_status": "transcript_error",
            "transcript_language": "",
            "transcript": "",
        }


def video_row(
    keyword: str,
    video_id: str,
    item: Dict,
    channel_details: Optional[Dict],
    transcript_details: Dict[str, str],
    comments_text_full: str,
) -> Dict[str, object]:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    channel_statistics = (channel_details or {}).get("statistics", {})

    return {
        "keyword": keyword,
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "channel_subscriber_count": channel_statistics.get("subscriberCount", 0),
        "published_at": snippet.get("publishedAt", ""),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "duration": content_details.get("duration", ""),
        "category_id": snippet.get("categoryId", ""),
        "tags": "|".join(snippet.get("tags", [])),
        "view_count": statistics.get("viewCount", 0),
        "like_count": statistics.get("likeCount", 0),
        "favorite_count": statistics.get("favoriteCount", 0),
        "comment_count": statistics.get("commentCount", 0),
        "definition": content_details.get("definition", ""),
        "caption": content_details.get("caption", ""),
        "licensed_content": content_details.get("licensedContent", ""),
        "transcript_status": transcript_details.get("transcript_status", ""),
        "transcript_language": transcript_details.get("transcript_language", ""),
        "transcript": transcript_details.get("transcript", ""),
        "share_count": "not_available",
        "comments_text_full": comments_text_full,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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


def parse_keywords(raw_keywords: Optional[str], keyword_file: Optional[Path]) -> List[str]:
    keywords = []

    if raw_keywords:
        keywords.extend(term.strip() for term in raw_keywords.split(","))

    if keyword_file:
        keywords.extend(
            line.strip()
            for line in keyword_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    if not keywords:
        keywords = DEFAULT_KEYWORDS

    return list(dict.fromkeys(term for term in keywords if term))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YouTube video data by pharma keywords.")
    parser.add_argument("--keywords", help="Comma-separated keywords. Defaults to pharma keyword list.")
    parser.add_argument("--keyword-file", type=Path, help="Text file with one keyword per line.")
    parser.add_argument("--output", default="youtube_video_data.csv", help="Video CSV output path.")
    parser.add_argument("--comments-output", default="youtube_comment_data.csv", help="Comment CSV output path.")
    parser.add_argument("--max-pages", type=int, default=2, help="Search pages per keyword. Each page has up to 50 videos.")
    parser.add_argument("--comment-pages", type=int, default=0, help="Comment pages per video. 0 disables comments.")
    parser.add_argument("--published-after", help="RFC3339 date, for example 2026-01-01T00:00:00Z.")
    parser.add_argument("--published-before", help="RFC3339 date, for example 2026-05-13T00:00:00Z.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--skip-transcripts", action="store_true", help="Disable transcript extraction.")
    parser.add_argument("--incremental", action="store_true", help="Append only new videos to existing output instead of overwriting.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set YOUTUBE_API_KEY in .env or PowerShell first: "
            "$env:YOUTUBE_API_KEY='YOUR_API_KEY'"
        )

    keywords = parse_keywords(args.keywords, args.keyword_file)
    print(f"Using {len(keywords)} keywords")
    if not args.skip_transcripts and YouTubeTranscriptApi is None:
        print("Transcript extraction disabled in practice: install youtube-transcript-api to populate transcript fields.")

    seen_video_ids = set()
    output_path = Path(args.output)
    comments_path = Path(args.comments_output)
    total_videos_saved = 0
    quota_exhausted = False

    if args.incremental and output_path.exists():
        existing = read_csv(output_path)
        seen_video_ids = {row["video_id"] for row in existing if row.get("video_id")}
        print(f"Incremental mode: {len(seen_video_ids)} existing video IDs loaded")

    for keyword in keywords:
        print(f"\nSearching YouTube for: {keyword}")
        try:
            video_ids = search_video_ids(
                keyword=keyword,
                api_key=api_key,
                max_pages=args.max_pages,
                published_after=args.published_after,
                published_before=args.published_before,
                sleep_seconds=args.sleep,
            )
        except RuntimeError as exc:
            if "quotaExceeded" in str(exc):
                print(f"\nQuota exhausted after keyword '{keyword}'. Saved {total_videos_saved} videos so far.")
                quota_exhausted = True
                break
            raise

        video_ids = [video_id for video_id in video_ids if video_id not in seen_video_ids]
        seen_video_ids.update(video_ids)

        print(f"Found {len(video_ids)} new videos")
        details = fetch_video_details(video_ids, api_key, args.sleep)
        channel_ids = list(
            dict.fromkeys(
                item.get("snippet", {}).get("channelId", "")
                for item in details.values()
                if item.get("snippet", {}).get("channelId", "")
            )
        )
        channel_details = fetch_channel_details(channel_ids, api_key, args.sleep)

        keyword_rows = []
        keyword_comment_rows = []
        for video_id, item in details.items():
            channel_id = item.get("snippet", {}).get("channelId", "")
            transcript_details = {
                "transcript_status": "transcript_skipped",
                "transcript_language": "",
                "transcript": "",
            }
            if not args.skip_transcripts:
                transcript_details = fetch_english_transcript(video_id)

            comments = []
            comments_text_full = ""
            if args.comment_pages > 0:
                comments = fetch_comments(
                    keyword=keyword,
                    video_id=video_id,
                    api_key=api_key,
                    max_pages=args.comment_pages,
                    sleep_seconds=args.sleep,
                )
                keyword_comment_rows.extend(comments)
                comments_text_full = build_comments_text_full(comments)

            keyword_rows.append(
                video_row(
                    keyword,
                    video_id,
                    item,
                    channel_details.get(channel_id),
                    transcript_details,
                    comments_text_full,
                )
            )

        # Flush to disk after every keyword so quota errors don't lose data
        if keyword_rows:
            append_csv(output_path, keyword_rows, VIDEO_COLUMNS)
            total_videos_saved += len(keyword_rows)
            print(f"  Saved {len(keyword_rows)} videos (total so far: {total_videos_saved})")

        if args.comment_pages > 0 and keyword_comment_rows:
            append_csv(comments_path, keyword_comment_rows, COMMENT_COLUMNS)

    if quota_exhausted:
        raise SystemExit(1)

    print(f"\nDone. Total new videos saved: {total_videos_saved} → {args.output}")


if __name__ == "__main__":
    main()
