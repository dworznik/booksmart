"""Unit tests for extraction response parsing and chapter slicing."""

import json

import pytest

from booksmart_core.extraction import (
    KNOWLEDGE_OBJECT_TYPES,
    REQUIRED_FIELDS,
    ExtractionError,
    chapter_body,
    parse_extraction_response,
)

VALID_ITEM = {
    "type": "Principle",
    "title": "Deep modules",
    "content": "Modules should be deep: simple interfaces over powerful functionality.",
    "summary": "Prefer deep modules.",
    "confidence": 0.9,
    "section_index": 0,
    "page": 4,
    "paragraph": 2,
}


def as_json(items: list[dict[str, object]]) -> str:
    return json.dumps(items)


class TestParseExtractionResponse:
    def test_parses_plain_json_array(self) -> None:
        objects, dropped = parse_extraction_response(as_json([VALID_ITEM]))

        assert dropped == []
        assert len(objects) == 1
        extracted = objects[0]
        assert extracted.type == "Principle"
        assert extracted.title == "Deep modules"
        assert extracted.confidence == 0.9
        assert extracted.section_index == 0
        assert extracted.page == 4
        assert extracted.paragraph == 2

    def test_strips_markdown_code_fences(self) -> None:
        fenced = "```json\n" + as_json([VALID_ITEM]) + "\n```"

        objects, dropped = parse_extraction_response(fenced)

        assert len(objects) == 1
        assert dropped == []

    def test_all_nine_types_are_accepted(self) -> None:
        items = [dict(VALID_ITEM, type=t) for t in sorted(KNOWLEDGE_OBJECT_TYPES)]

        objects, dropped = parse_extraction_response(as_json(items))

        assert {o.type for o in objects} == KNOWLEDGE_OBJECT_TYPES
        assert dropped == []

    def test_unknown_type_drops_element_and_keeps_the_rest(self) -> None:
        # Models sometimes reach for the book's own vocabulary ("Red Flag");
        # one mislabeled element must not cost the whole response.
        items = [VALID_ITEM, dict(VALID_ITEM, type="Red Flag")]

        objects, dropped = parse_extraction_response(as_json(items))

        assert len(objects) == 1
        assert objects[0].type == "Principle"
        assert len(dropped) == 1
        assert "Red Flag" in dropped[0]

    def test_missing_required_field_drops_element_with_reason(self) -> None:
        item = dict(VALID_ITEM)
        del item["summary"]

        objects, dropped = parse_extraction_response(as_json([item, VALID_ITEM]))

        assert len(objects) == 1
        assert len(dropped) == 1
        assert "summary" in dropped[0]

    def test_non_object_element_drops_with_reason(self) -> None:
        objects, dropped = parse_extraction_response(as_json([VALID_ITEM, "just a string"]))  # type: ignore[list-item]

        assert len(objects) == 1
        assert len(dropped) == 1
        assert "element 1" in dropped[0]

    def test_non_list_payload_is_rejected(self) -> None:
        with pytest.raises(ExtractionError, match="array"):
            parse_extraction_response('{"type": "Principle"}')

    def test_invalid_json_is_rejected(self) -> None:
        with pytest.raises(ExtractionError):
            parse_extraction_response("the model rambled instead of emitting JSON")

    def test_optional_fields_default_to_none(self) -> None:
        item = {k: v for k, v in VALID_ITEM.items() if k in REQUIRED_FIELDS}

        objects, _ = parse_extraction_response(as_json([item]))
        extracted = objects[0]

        assert extracted.section_index is None
        assert extracted.page is None
        assert extracted.paragraph is None


class TestChapterBody:
    MARKDOWN = "\n".join(
        [
            "# Chapter One",  # line 1
            "First chapter text.",
            "## Section A",
            "Section A text.",
            "# Chapter Two",  # line 5
            "Second chapter text.",
        ]
    )

    def test_slices_from_heading_to_next_chapter(self) -> None:
        body = chapter_body(self.MARKDOWN, start_line=1, next_start_line=5)

        assert "First chapter text." in body
        assert "Section A text." in body
        assert "Second chapter text." not in body

    def test_last_chapter_runs_to_end_of_document(self) -> None:
        body = chapter_body(self.MARKDOWN, start_line=5, next_start_line=None)

        assert "Second chapter text." in body
        assert "First chapter text." not in body

    def test_missing_start_line_falls_back_to_whole_document(self) -> None:
        body = chapter_body(self.MARKDOWN, start_line=None, next_start_line=None)

        assert body == self.MARKDOWN
