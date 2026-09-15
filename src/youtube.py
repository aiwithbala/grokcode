"""Detect YouTube URLs and fetch captions/transcripts when available."""

from __future__ import annotations

import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi

try:
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApiException,
    )
except ImportError:  # pragma: no cover — older / alternate package layouts
    from youtube_transcript_api._errors import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApiException,
    )

# Soft cap for transcript text sent to the model.
MAX_TRANSCRIPT_CHARS = 80_000

_YOUTUBE_RE = re.compile(
    r"(?:"
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?[^\s]*\bv=([\w-]{11})"
    r"|(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]{11})"
    r"|(?:https?://)?(?:www\.)?youtu\.be/([\w-]{11})"
    r"|(?:https?://)?(?:www\.)?youtube\.com/embed/([\w-]{11})"
    r")",
    re.IGNORECASE,
)


@dataclass
class YouTubeTranscriptResult:
    url: str
    video_id: str
    transcript: str = ""
    language: str | None = None
    error: str | None = None


def extract_youtube_ids(text: str) -> list[tuple[str, str]]:
    """Return unique (matched_url_fragment, video_id) pairs found in text."""
    if not text:
        return []
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _YOUTUBE_RE.finditer(text):
        video_id = next(g for g in match.groups() if g)
        if video_id in seen:
            continue
        seen.add(video_id)
        found.append((match.group(0), video_id))
    return found


def _truncate(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    note = (
        f"\n\n[Note: transcript truncated from {len(text):,} to {limit:,} characters "
        "to stay within context limits.]"
    )
    keep = max(0, limit - len(note))
    return text[:keep] + note


def fetch_transcript(video_id: str, url_hint: str = "") -> YouTubeTranscriptResult:
    """Fetch captions for a YouTube video ID. Friendly error if unavailable."""
    url = url_hint or f"https://www.youtube.com/watch?v={video_id}"
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en", "en-US", "en-GB"])
        # Prefer to_raw_data when available; fall back to snippets.
        if hasattr(fetched, "to_raw_data"):
            rows = fetched.to_raw_data()
            lines = [row.get("text", "") for row in rows if row.get("text")]
        else:
            snippets = getattr(fetched, "snippets", None) or fetched
            lines = []
            for snip in snippets:
                text = getattr(snip, "text", None)
                if text is None and isinstance(snip, dict):
                    text = snip.get("text", "")
                if text:
                    lines.append(text)
        body = " ".join(lines).strip()
        if not body:
            return YouTubeTranscriptResult(
                url=url,
                video_id=video_id,
                error=f"No caption text was returned for YouTube video {video_id}.",
            )
        language = getattr(fetched, "language_code", None) or getattr(
            fetched, "language", None
        )
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            transcript=_truncate(body),
            language=str(language) if language else None,
        )
    except TranscriptsDisabled:
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            error=(
                f"Captions are disabled for this YouTube video ({video_id}). "
                "I can't fetch a transcript."
            ),
        )
    except NoTranscriptFound:
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            error=(
                f"No captions/transcript are available for this YouTube video ({video_id})."
            ),
        )
    except VideoUnavailable:
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            error=f"YouTube video {video_id} is unavailable or private.",
        )
    except YouTubeTranscriptApiException as exc:
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            error=f"Could not fetch YouTube captions for {video_id}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return YouTubeTranscriptResult(
            url=url,
            video_id=video_id,
            error=f"Could not fetch YouTube captions for {video_id}: {exc}",
        )


def gather_transcripts_from_text(text: str) -> list[YouTubeTranscriptResult]:
    """Detect YouTube URLs in text and fetch each transcript."""
    results: list[YouTubeTranscriptResult] = []
    for url_frag, video_id in extract_youtube_ids(text):
        results.append(fetch_transcript(video_id, url_hint=url_frag))
    return results


def format_transcripts_for_prompt(results: list[YouTubeTranscriptResult]) -> str:
    """Render successful transcripts as a context block."""
    blocks: list[str] = []
    for r in results:
        if r.error or not r.transcript:
            continue
        lang = f" ({r.language})" if r.language else ""
        blocks.append(
            f"### YouTube transcript{lang}\n"
            f"Source: {r.url}\n\n"
            f"{r.transcript}"
        )
    if not blocks:
        return ""
    return (
        "The user referenced the following YouTube video(s). "
        "Use the transcript(s) to answer the instruction.\n\n"
        + "\n\n".join(blocks)
    )
