"""Knowledge object extraction from parsed chapter text.

The LLM must return a strict JSON array; anything else raises ExtractionError
and fails the run rather than persisting objects with broken provenance.
Bump EXTRACTION_PROMPT_VERSION whenever the prompt wording changes so stored
objects record exactly what produced them.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from app.llm import strip_fences
from app.models import Book, Chapter, Section

EXTRACTION_PROMPT_VERSION = "1"

KnowledgeType = Literal[
    "Practice",
    "Principle",
    "Tradeoff",
    "Anti-pattern",
    "Smell",
    "Decision Rule",
    "Definition",
    "Glossary",
    "Checklist",
]

KNOWLEDGE_OBJECT_TYPES: frozenset[str] = frozenset(get_args(KnowledgeType))

EXTRACTION_SYSTEM_PROMPT = (
    "You extract candidate knowledge objects from technical book chapters for "
    "a knowledge repository. Respond with a JSON array only - no prose, no "
    "markdown fences. Each element must be an object with fields: "
    '"type" (one of: ' + ", ".join(sorted(KNOWLEDGE_OBJECT_TYPES)) + "), "
    '"title" (a short name), "content" (the full idea in the book\'s own terms), '
    '"summary" (one sentence), "confidence" (number from 0.0 to 1.0), '
    '"section_index" (0-based index into the numbered section list, or null when '
    "the idea is not tied to one section), "
    '"page" (integer, only when the text carries an explicit page marker, else null), '
    '"paragraph" (integer, only when the paragraph is unambiguous, else null). '
    "Never guess page or paragraph numbers. Extract only ideas the chapter "
    "actually asserts; return [] for a chapter with none."
)

REQUIRED_FIELDS = ("type", "title", "content", "summary", "confidence")


class ExtractionError(RuntimeError):
    """The LLM response could not be turned into valid knowledge objects."""


@dataclass(frozen=True)
class ExtractedObject:
    type: str
    title: str
    content: str
    summary: str
    confidence: float
    section_index: int | None
    page: int | None
    paragraph: int | None


def _optional_int(item: dict[str, object], field: str, position: int) -> int | None:
    value = item.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractionError(f"element {position}: {field!r} must be an integer or null")
    return value


def parse_extraction_response(text: str) -> list[ExtractedObject]:
    payload = strip_fences(text.strip())
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ExtractionError("response must be a JSON array of knowledge objects")

    objects: list[ExtractedObject] = []
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise ExtractionError(f"element {position} is not a JSON object")
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ExtractionError(f"element {position} is missing {field!r}")
        if item["type"] not in KNOWLEDGE_OBJECT_TYPES:
            raise ExtractionError(f"element {position} has unsupported type {item['type']!r}")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ExtractionError(f"element {position}: 'confidence' must be a number")
        objects.append(
            ExtractedObject(
                type=str(item["type"]),
                title=str(item["title"]),
                content=str(item["content"]),
                summary=str(item["summary"]),
                confidence=float(confidence),
                section_index=_optional_int(item, "section_index", position),
                page=_optional_int(item, "page", position),
                paragraph=_optional_int(item, "paragraph", position),
            )
        )
    return objects


def chapter_body(markdown: str, start_line: int | None, next_start_line: int | None) -> str:
    """The chapter's slice of the parsed markdown, heading line included.

    Line numbers are 1-based, as recorded by structure detection. A chapter
    without a recorded start falls back to the whole document.
    """
    if start_line is None:
        return markdown
    lines = markdown.splitlines()
    end = next_start_line - 1 if next_start_line is not None else len(lines)
    return "\n".join(lines[start_line - 1 : end])


def iter_chapter_bodies(
    chapters: Sequence[Chapter], markdown: str
) -> Iterator[tuple[Chapter, str]]:
    """Each chapter paired with its slice of the parsed markdown."""
    for index, chapter in enumerate(chapters):
        next_chapter = chapters[index + 1] if index + 1 < len(chapters) else None
        yield (
            chapter,
            chapter_body(
                markdown, chapter.source_line, next_chapter.source_line if next_chapter else None
            ),
        )


def resolve_source(chapter: Chapter, section_index: int | None) -> tuple[Section | None, str]:
    """The section the LLM pointed at (None when absent or out of range) and the
    human-readable source location string."""
    section = None
    if section_index is not None and 0 <= section_index < len(chapter.sections):
        section = chapter.sections[section_index]
    source_location = f"chapter {chapter.position + 1}: {chapter.title}"
    if section is not None:
        source_location += f" > {section.title}"
    return section, source_location


def build_chapter_prompt(book: Book, chapter: Chapter, body: str, instruction: str) -> str:
    """Shared prompt scaffold for per-chapter stages: book/chapter header, the
    numbered section list (indexes matter - both extraction's section_index and
    the summary stage's section alignment refer to it), the chapter text, and
    the stage's closing instruction."""
    lines = [
        f"Book: {book.title} by {book.author}",
        f"Chapter {chapter.position + 1}: {chapter.title}",
        "",
        "Numbered sections in this chapter:",
    ]
    if chapter.sections:
        lines.extend(f"{index}. {section.title}" for index, section in enumerate(chapter.sections))
    else:
        lines.append("(none detected)")
    lines.extend(["", "Chapter text:", body, "", instruction])
    return "\n".join(lines)


def build_extraction_prompt(book: Book, chapter: Chapter, body: str) -> str:
    return build_chapter_prompt(book, chapter, body, "Extract the knowledge objects as JSON.")
