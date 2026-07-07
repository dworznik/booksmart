"""Unit tests for chapter/section summary response parsing."""

import json

import pytest

from app.summaries import SummaryError, parse_summary_response

VALID = {
    "chapter_summary": "Modules should be deep.",
    "section_summaries": ["About deep modules.", "About shallow modules."],
}


class TestParseSummaryResponse:
    def test_parses_summaries(self) -> None:
        chapter_summary, section_summaries = parse_summary_response(
            json.dumps(VALID), section_count=2
        )

        assert chapter_summary == "Modules should be deep."
        assert section_summaries == ["About deep modules.", "About shallow modules."]

    def test_strips_markdown_code_fences(self) -> None:
        fenced = "```json\n" + json.dumps(VALID) + "\n```"

        chapter_summary, _ = parse_summary_response(fenced, section_count=2)

        assert chapter_summary == "Modules should be deep."

    def test_short_section_list_is_padded_with_none(self) -> None:
        payload = dict(VALID, section_summaries=["Only the first."])

        _, section_summaries = parse_summary_response(json.dumps(payload), section_count=3)

        assert section_summaries == ["Only the first.", None, None]

    def test_long_section_list_is_truncated(self) -> None:
        payload = dict(VALID, section_summaries=["a", "b", "c", "d"])

        _, section_summaries = parse_summary_response(json.dumps(payload), section_count=2)

        assert section_summaries == ["a", "b"]

    def test_missing_chapter_summary_is_rejected(self) -> None:
        with pytest.raises(SummaryError, match="chapter_summary"):
            parse_summary_response('{"section_summaries": []}', section_count=0)

    def test_non_object_payload_is_rejected(self) -> None:
        with pytest.raises(SummaryError, match="object"):
            parse_summary_response('["not", "an", "object"]', section_count=0)

    def test_invalid_json_is_rejected(self) -> None:
        with pytest.raises(SummaryError):
            parse_summary_response("prose instead of JSON", section_count=0)

    def test_null_section_entries_stay_none(self) -> None:
        payload = dict(VALID, section_summaries=[None, "Second."])

        _, section_summaries = parse_summary_response(json.dumps(payload), section_count=2)

        assert section_summaries == [None, "Second."]
