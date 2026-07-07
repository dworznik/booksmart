"""Chapter and section summary generation, feeding the embedding stage.

The PRD stores embeddings of chapter and section summaries in Qdrant but no
earlier stage produces summaries, so this one does: one LLM call per chapter
over the chapter's slice of the parsed markdown. Bump SUMMARY_PROMPT_VERSION
whenever the prompt wording changes.
"""

import json

from app.errors import ProviderResponseError
from app.extraction import build_chapter_prompt
from app.llm import strip_fences
from app.models import Book, Chapter

SUMMARY_PROMPT_VERSION = "1"

SUMMARY_SYSTEM_PROMPT = (
    "You summarize technical book chapters for a knowledge repository. "
    "Respond with a JSON object only - no prose, no markdown fences - with "
    'fields: "chapter_summary" (2-4 sentences covering what the chapter '
    'teaches) and "section_summaries" (an array with exactly one entry per '
    "numbered section, in order: 1-2 sentences each, or null when a section "
    "has no summarizable content)."
)


class SummaryError(ProviderResponseError):
    """The LLM response could not be turned into usable summaries. A retriable
    ProviderResponseError: when it escapes a stage (unparseable even after the
    stage's own retry), a fresh attempt may still succeed."""


def parse_summary_response(text: str, section_count: int) -> tuple[str, list[str | None]]:
    """The chapter summary and per-section summaries, aligned to section_count.

    Models occasionally miscount sections; a short list is padded with None
    and a long one truncated, so alignment errors degrade a section's summary
    rather than failing the run.
    """
    payload = strip_fences(text.strip())
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SummaryError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError("response must be a JSON object")
    chapter_summary = data.get("chapter_summary")
    if not isinstance(chapter_summary, str) or not chapter_summary.strip():
        raise SummaryError("response is missing 'chapter_summary'")
    raw_sections = data.get("section_summaries")
    if raw_sections is None:
        raw_sections = []
    if not isinstance(raw_sections, list):
        raise SummaryError("'section_summaries' must be an array")

    section_summaries: list[str | None] = []
    for entry in raw_sections[:section_count]:
        section_summaries.append(entry if isinstance(entry, str) and entry.strip() else None)
    section_summaries.extend([None] * (section_count - len(section_summaries)))
    return chapter_summary, section_summaries


def build_summary_prompt(book: Book, chapter: Chapter, body: str) -> str:
    return build_chapter_prompt(
        book, chapter, body, "Summarize the chapter and its sections as JSON."
    )
