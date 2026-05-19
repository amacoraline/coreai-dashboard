import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import RequestError
except ImportError:
    GoogleTranslator = None
    RequestError = Exception


TEXT_COLUMNS_TO_TRANSLATE = [
    "keyword",
    "title",
    "description",
    "channel_title",
    "transcript",
    "comments_text_full",
]


TRANSLATION_CACHE = {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def iso8601_duration_to_seconds(value: str) -> int:
    if not value:
        return 0

    match = re.fullmatch(
        r"P(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)",
        value.strip(),
    )

    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_duration_text(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def build_comments_lookup(comment_rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    comments_lookup: Dict[str, List[str]] = {}

    for row in comment_rows:
        video_id = row.get("video_id", "").strip()
        text = row.get("text", "").strip()

        if not video_id or not text:
            continue

        comments_lookup.setdefault(video_id, []).append(text)

    return comments_lookup


def safe_translate(
    text: str,
    translator: Optional[GoogleTranslator],
    retries: int = 3,
    sleep_time: float = 1.0,
) -> str:
    """
    Safe translation with:
    - retries
    - caching
    - throttling
    - graceful fallback
    """

    value = (text or "").strip()

    if not value:
        return ""

    if translator is None:
        return value

    # Cache hit
    if value in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[value]

    max_chunk_size = 4000

    parts = [
        value[index:index + max_chunk_size]
        for index in range(0, len(value), max_chunk_size)
    ]

    translated_parts = []

    for part in parts:

        translated_text = part

        for attempt in range(retries):
            try:
                translated = translator.translate(part)

                if isinstance(translated, str) and translated.strip():
                    translated_text = translated.strip()

                break

            except RequestError as error:
                print(
                    f"[Retry {attempt + 1}/{retries}] "
                    f"Translation failed: {str(error)}"
                )

                time.sleep(sleep_time * (attempt + 1))

            except Exception as error:
                print(f"[Unexpected Translation Error] {str(error)}")
                break

        translated_parts.append(translated_text)

        # Throttle requests to avoid Google blocking
        time.sleep(0.5)

    final_text = "\n".join(translated_parts)

    TRANSLATION_CACHE[value] = final_text

    return final_text


def clean_row(
    row: Dict[str, str],
    comments_lookup: Dict[str, List[str]],
    translator: Optional[GoogleTranslator],
) -> Dict[str, object]:

    cleaned = dict(row)

    available_columns = set(row.keys())

    video_id = row.get("video_id", "").strip()

    existing_comments_text = row.get("comments_text_full", "").strip()

    if not existing_comments_text and video_id in comments_lookup:
        existing_comments_text = "\n\n".join(comments_lookup[video_id])

    duration_raw = row.get("duration", "").strip()

    duration_seconds = iso8601_duration_to_seconds(duration_raw)

    duration_minutes = (
        round(duration_seconds / 60, 2)
        if duration_seconds
        else 0
    )

    cleaned["comments_text_full"] = existing_comments_text
    cleaned["duration_seconds"] = duration_seconds
    cleaned["duration_minutes"] = duration_minutes

    cleaned["duration_text"] = (
        seconds_to_duration_text(duration_seconds)
        if duration_seconds
        else ""
    )

    for column in TEXT_COLUMNS_TO_TRANSLATE:

        if column not in available_columns:
            continue

        source_value = cleaned.get(column, "")

        try:
            cleaned[f"{column}_en"] = safe_translate(
                source_value,
                translator,
            )

        except Exception as error:
            print(
                f"[Translation Failed] "
                f"Column={column} "
                f"Error={str(error)}"
            )

            cleaned[f"{column}_en"] = source_value

    cleaned["analysis_text_en"] = "\n".join(
        part
        for part in [
            cleaned.get("keyword_en", ""),
            cleaned.get("title_en", ""),
            cleaned.get("description_en", ""),
            cleaned.get("transcript_en", ""),
            cleaned.get("comments_text_full_en", ""),
            cleaned.get("channel_title_en", ""),
        ]
        if part
    ).strip()

    return cleaned


def build_output_columns(base_columns: List[str]) -> List[str]:

    ordered_columns: List[str] = []

    for column in base_columns:

        ordered_columns.append(column)

        if column in TEXT_COLUMNS_TO_TRANSLATE:
            ordered_columns.append(f"{column}_en")

    extra_columns = [
        "duration_seconds",
        "duration_minutes",
        "duration_text",
        "analysis_text_en",
    ]

    for extra_column in extra_columns:

        if extra_column not in ordered_columns:
            ordered_columns.append(extra_column)

    return ordered_columns


def default_output_path(input_path: Path) -> Path:

    if input_path.stem.endswith("_cleaned"):
        return input_path

    return input_path.with_name(
        f"{input_path.stem}_cleaned{input_path.suffix}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Clean and translate extracted YouTube data."
    )

    parser.add_argument(
        "--input",
        default="youtube_video_data.csv",
        help="Raw video dataset.",
    )

    parser.add_argument(
        "--comments-input",
        default="youtube_comment_data.csv",
        help="Optional comments dataset.",
    )

    parser.add_argument(
        "--output",
        help="Cleaned dataset output.",
    )

    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation and only normalize fields.",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip rows already present in the output file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    video_rows = read_csv(input_path)

    if not video_rows:
        raise SystemExit(f"No rows found in {args.input}")

    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(input_path)
    )

    if args.incremental and output_path.exists():
        existing_rows = read_csv(output_path)
        processed_ids = {row.get("video_id", "") for row in existing_rows if row.get("video_id")}
        video_rows = [row for row in video_rows if row.get("video_id", "") not in processed_ids]
        print(f"Incremental mode: {len(video_rows)} new rows to clean (skipped {len(processed_ids)} already processed)")

    if not video_rows:
        print("\nNo new rows to clean.")
        return

    comment_rows = read_csv(Path(args.comments_input))

    comments_lookup = build_comments_lookup(comment_rows)

    translator = None

    if not args.skip_translation:

        if GoogleTranslator is None:
            raise SystemExit(
                "Install dependency first:\n"
                "pip install deep-translator"
            )

        translator = GoogleTranslator(
            source="auto",
            target="en",
        )

    cleaned_rows = []

    total_rows = len(video_rows)

    for index, row in enumerate(video_rows, start=1):

        print(f"Processing row {index}/{total_rows}")

        cleaned_row_data = clean_row(
            row,
            comments_lookup,
            translator,
        )

        cleaned_rows.append(cleaned_row_data)

    output_columns = build_output_columns(
        list(video_rows[0].keys())
    )

    if args.incremental and output_path.exists():
        with output_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=output_columns)
            writer.writerows(cleaned_rows)
        print(f"\nAppended {len(cleaned_rows)} rows to {output_path}")
    else:
        write_csv(
            output_path,
            cleaned_rows,
            output_columns,
        )
        print(f"\nSaved {len(cleaned_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()