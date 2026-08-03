"""Loading a truth tree, and the guards that keep it honest.

Ported from the hybrid-eval fixture guards: a query set that lies about its own
kinds turns a benchmark into a measurement of nothing, and unlike a fixture in
this repo, truth is authored by hand in another one — so the check has to travel
with the harness that reads it.

Every fixture here is synthetic. Real titles and real corpus shape live in the
private assets repo, never in these tests.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from booksmart_bench.errors import BenchError
from booksmart_bench.truth import content_words, errors_in, load_truth, lint


def messages(findings: object) -> str:
    return "\n".join(f"{f.severity}: {f.where}: {f.message}" for f in findings)  # type: ignore[attr-defined]


class TestLoad:
    def test_reads_a_books_identity_and_structure(
        self, write_truth: Callable[..., Path]
    ) -> None:
        truth = load_truth(write_truth("a-book"))

        book = truth.books["a-book"]
        assert book.title == "Placeholder Book"
        assert book.area == "placeholder-area"
        assert set(book.nodes) == {"1", "1.1", "1.2"}

    def test_sections_know_their_chapter(self, write_truth: Callable[..., Path]) -> None:
        truth = load_truth(write_truth("a-book"))

        assert truth.books["a-book"].nodes["1.1"].parent == "1"

    def test_front_and_back_matter_become_nodes_too(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A query may legitimately expect a preface or an appendix."""
        assets = write_truth(
            "a-book",
            toc={
                "front_matter": [{"id": "foreword", "title": "Foreword"}],
                "chapters": [{"id": "1", "title": "First Chapter"}],
                "back_matter": [{"id": "index", "title": "Index"}],
            },
        )

        assert set(load_truth(assets).books["a-book"].nodes) == {"foreword", "1", "index"}

    def test_area_queries_are_loaded_separately(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            area="an-area",
            area_queries=[
                {
                    "q": "a cross-book question",
                    "kind": "conceptual",
                    "expects": [{"book": "a-book", "loc": "1.1"}],
                    "why": "Because.",
                }
            ],
        )

        assert len(load_truth(assets).areas["an-area"]) == 1

    def test_a_truth_dir_missing_its_toc_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        (assets / "truth" / "a-book" / "toc.yaml").unlink()

        with pytest.raises(BenchError, match="toc.yaml"):
            load_truth(assets)

    def test_malformed_yaml_names_the_file(self, write_truth: Callable[..., Path]) -> None:
        assets = write_truth("a-book")
        (assets / "truth" / "a-book" / "queries.yaml").write_text("- q: [unclosed\n")

        with pytest.raises(BenchError, match="queries.yaml"):
            load_truth(assets)


class TestLocationIntegrity:
    def test_a_query_expecting_an_unknown_node_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """The one failure that silently zeroes a score: an expectation nothing
        can ever satisfy."""
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "grommets",
                    "kind": "exact-term",
                    "expects": [{"loc": "9.9"}],
                    "why": "Typo'd location.",
                }
            ],
        )

        findings = errors_in(lint(load_truth(assets)))

        assert "9.9" in messages(findings)

    def test_a_concept_expecting_an_unknown_node_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book", concepts=[{"concept": "Some idea", "loc": "nowhere"}]
        )

        assert "nowhere" in messages(errors_in(lint(load_truth(assets))))

    def test_an_index_pair_expecting_an_unknown_node_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book", index_pairs=[{"term": "widget", "loc": "nowhere"}])

        assert "nowhere" in messages(errors_in(lint(load_truth(assets))))

    def test_a_duplicate_node_id_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Ids are what every other file references; two nodes sharing one makes
        every reference to it ambiguous."""
        assets = write_truth(
            "a-book",
            toc={
                "chapters": [
                    {"id": "1", "title": "First", "sections": [{"id": "1.1", "title": "A"}]},
                    {"id": "1", "title": "Second"},
                ]
            },
        )

        assert "duplicate" in messages(errors_in(lint(load_truth(assets)))).lower()

    def test_an_area_query_naming_an_unknown_book_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            area="an-area",
            area_queries=[
                {
                    "q": "a question",
                    "kind": "conceptual",
                    "expects": [{"book": "no-such-book", "loc": "1.1"}],
                    "why": "Because.",
                }
            ],
        )

        assert "no-such-book" in messages(errors_in(lint(load_truth(assets))))

    def test_an_unauthored_placeholder_is_a_warning_not_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Area queries are authored before every book in the area is, so a TBD
        must be visible without failing the tree it sits in."""
        assets = write_truth(
            "a-book",
            area="an-area",
            area_queries=[
                {
                    "q": "a question",
                    "kind": "conceptual",
                    "expects": [{"book": "not-yet-authored", "loc": "TBD"}],
                    "why": "Because.",
                }
            ],
        )

        findings = lint(load_truth(assets))

        assert not errors_in(findings)
        assert "TBD" in messages(findings)


class TestQueryKindGuards:
    def test_a_conceptual_query_sharing_vocabulary_with_its_target_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """The defining property of the slice: if the words are shared, lexical
        retrieval can find it and the query is not conceptual at all."""
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "how do widgets relate to sprockets",
                    "kind": "conceptual",
                    "expects": [{"loc": "1.1"}],  # "Widgets And Sprockets"
                    "why": "Mislabelled — this is exact-term vocabulary.",
                }
            ],
        )

        found = messages(errors_in(lint(load_truth(assets))))
        assert "widget" in found and "sprocket" in found

    def test_a_conceptual_query_avoiding_its_targets_vocabulary_passes(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "the small toothed parts that mesh to transmit turning force",
                    "kind": "conceptual",
                    "expects": [{"loc": "1.1"}],
                    "why": "A paraphrase with none of the section's own words.",
                }
            ],
        )

        assert not errors_in(lint(load_truth(assets)))

    def test_stemming_catches_a_shared_word_in_another_form(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Lexical retrieval stems, so the guard has to as well — "grommet"
        against a section called "Grommets" is a shared word."""
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "what is a grommet used for",
                    "kind": "conceptual",
                    "expects": [{"loc": "1.2"}],  # "Grommets"
                    "why": "Shares a stem with the title.",
                }
            ],
        )

        assert "grommet" in messages(errors_in(lint(load_truth(assets))))

    def test_an_unknown_kind_is_an_error(self, write_truth: Callable[..., Path]) -> None:
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "something",
                    "kind": "vibes",
                    "expects": [{"loc": "1.1"}],
                    "why": "Not one of the three.",
                }
            ],
        )

        assert "vibes" in messages(errors_in(lint(load_truth(assets))))

    def test_a_query_without_a_rationale_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """The why-line is what stops a query set drifting into whatever the
        pipeline currently happens to answer well."""
        assets = write_truth(
            "a-book",
            queries=[
                {"q": "something", "kind": "exact-term", "expects": [{"loc": "1.1"}], "why": "  "}
            ],
        )

        assert "why" in messages(errors_in(lint(load_truth(assets)))).lower()

    def test_a_query_expecting_nothing_is_an_error(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            queries=[{"q": "something", "kind": "exact-term", "expects": [], "why": "Why."}],
        )

        assert "expects" in messages(errors_in(lint(load_truth(assets))))

    def test_a_long_exact_term_query_is_a_warning(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """An exact-term query is meant to be the book's own coined phrase; a
        whole sentence is a mixed query wearing the wrong label."""
        assets = write_truth(
            "a-book",
            queries=[
                {
                    "q": "what exactly happens when a widget is connected to a sprocket assembly",
                    "kind": "exact-term",
                    "expects": [{"loc": "1.1"}],
                    "why": "Too long to be a term.",
                }
            ],
        )

        findings = lint(load_truth(assets))

        assert not errors_in(findings)
        assert "exact-term" in messages(findings)


class TestCoverageWarnings:
    def test_a_thin_query_set_warns_per_kind(self, write_truth: Callable[..., Path]) -> None:
        """Per-kind means are averaged over these; too few and one query's luck
        moves the score."""
        assets = write_truth(
            "a-book",
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
        )

        assert "conceptual" in messages(lint(load_truth(assets)))

    def test_an_unpinned_source_is_a_warning(self, write_truth: Callable[..., Path]) -> None:
        """Truth authored against an unpinned artifact can silently score a
        different edition."""
        assets = write_truth(
            "a-book",
            book=dict(
                _book_defaults(),
                slug="a-book",
                source={"file": "sources/a-book.pdf", "sha256": "TBD-when-file-lands"},
            ),
        )

        assert "sha256" in messages(lint(load_truth(assets)))


class TestContentWords:
    def test_ignores_short_words(self) -> None:
        assert content_words("a of the by an") == set()

    def test_folds_plural_and_singular_together(self) -> None:
        assert content_words("grommets") == content_words("grommet")

    def test_is_case_insensitive(self) -> None:
        assert content_words("Widgets") == content_words("widgets")


def _book_defaults() -> dict[str, object]:
    return {
        "title": "Placeholder Book",
        "edition": "1st",
        "authors": ["A. Author"],
        "area": "placeholder-area",
    }
