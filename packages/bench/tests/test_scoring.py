"""Run file × truth -> scores, and the regression verdict over two runs.

Every test here is pure: no pipeline, no corpus, no network. That is the point
of the artefact split — a scorer that needed a pipeline to test would be a
scorer that could not be trusted to compare two of them.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from booksmart_bench.errors import BenchError
from booksmart_bench.scoring import (
    THRESHOLDS,
    compare,
    load_run,
    score,
)
from booksmart_bench.truth import load_truth

from .conftest import hits


def scored(
    write_truth: Callable[..., Path],
    write_run: Callable[..., Path],
    *,
    queries: list[dict[str, object]] | None = None,
    **run_sections: object,
) -> object:
    """Author a truth tree and a run over it, and score the pair."""
    assets = write_truth("placeholder-book", queries=queries or [])
    run = load_run(write_run(**run_sections))
    return score(run, load_truth(assets))


class TestLoad:
    def test_reads_identity_and_results(self, write_run: Callable[..., Path]) -> None:
        run = load_run(write_run(run_id="r1", snapshot="snap", books=["a"]))

        assert (run.run_id, run.snapshot, run.books) == ("r1", "snap", ("a",))

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(BenchError, match=re.escape("nowhere.json")):
            load_run(tmp_path / "nowhere.json")

    def test_malformed_json_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")

        with pytest.raises(BenchError, match=re.escape("bad.json")):
            load_run(path)

    def test_reads_the_documented_shape(self, tmp_path: Path) -> None:
        """The shape the assets repo's SAMPLE run file documents. Written out
        here rather than read from that checkout: it names real books, and this
        repo is public."""
        path = tmp_path / "documented.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": "2026-01-01T00-00-00Z-recall-area",
                    "corpus": {"snapshot": "snap", "books": ["a-book"]},
                    "search": {
                        "mode": "hybrid",
                        "limit": 10,
                        "record_types": ["chapter", "section", "knowledge_object"],
                    },
                    "results": [
                        {
                            "q": "a coined term",
                            "book": "a-book",
                            "hits": [
                                {"rank": 1, "book": "a-book", "loc": "4.1", "record_type": "section"},
                                {"rank": 2, "book": "a-book", "loc": "4", "record_type": "chapter"},
                            ],
                        }
                    ],
                }
            )
        )

        run = load_run(path)

        assert run.search.mode == "hybrid"
        assert run.results[0].hits[0].loc == "4.1"
        assert run.results[0].hits[1].record_type == "chapter"


class TestMalformedRunFiles:
    def test_a_non_numeric_rank_is_a_bench_error_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        """The CLI renders BenchError as one line; anything else is a traceback
        in the face of someone whose only mistake was a hand-edited run file."""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": "r",
                    "corpus": {"snapshot": "s", "books": []},
                    "search": {},
                    "results": [{"q": "a", "hits": [{"book": "b", "loc": "1", "rank": "one"}]}],
                }
            )
        )

        with pytest.raises(BenchError, match=re.escape("bad.json")):
            load_run(path)

    def test_a_non_numeric_claim_count_is_a_bench_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": "r",
                    "corpus": {"snapshot": "s", "books": []},
                    "search": {},
                    "faithfulness": [{"book": "b", "loc": "1", "supported": 1, "claims": {}}],
                }
            )
        )

        with pytest.raises(BenchError, match=re.escape("bad.json")):
            load_run(path)


class TestRanks:
    """A rank is 1-based. Anything else is malformed, and the two metrics must
    at least agree about it."""

    @pytest.mark.parametrize("rank", [0, -1])
    def test_a_non_positive_rank_counts_as_a_miss_for_both_metrics(
        self,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
        rank: int,
    ) -> None:
        assets = write_truth(
            "placeholder-book",
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
        )
        run = load_run(
            write_run(
                results=[
                    {
                        "q": "grommets",
                        "hits": [
                            {"rank": rank, "book": "placeholder-book", "loc": "1.2"}
                        ],
                    }
                ]
            )
        )

        result = score(run, load_truth(assets))

        assert result.recall is not None
        assert result.recall.mrr == 0.0
        assert result.recall.hit_at_5 == 0.0


class TestRecall:
    """Binary location-level judgements: MRR primary, hit@5 alongside."""

    def test_a_hit_at_rank_one_scores_a_full_reciprocal(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[{"q": "grommets", "book": "placeholder-book", "hits": hits("1.2")}],
        )

        assert result.recall is not None
        assert result.recall.mrr == 1.0
        assert result.recall.hit_at_5 == 1.0

    def test_rank_four_scores_one_quarter_and_still_hits_at_five(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[
                {"q": "grommets", "book": "placeholder-book", "hits": hits("1", "1.1", "9", "1.2")}
            ],
        )

        assert result.recall is not None
        assert result.recall.mrr == pytest.approx(0.25)
        assert result.recall.hit_at_5 == 1.0

    def test_a_hit_below_rank_five_still_earns_mrr_but_not_hit_at_5(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[
                {
                    "q": "grommets",
                    "book": "placeholder-book",
                    "hits": hits("1", "1.1", "9", "8", "7", "1.2"),
                }
            ],
        )

        assert result.recall is not None
        assert result.recall.mrr == pytest.approx(1 / 6)
        assert result.recall.hit_at_5 == 0.0

    def test_no_hit_at_all_scores_zero_without_being_dropped(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """A query that found nothing must stay in the denominator, or the mean
        measures only the queries that happened to work."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[{"q": "grommets", "book": "placeholder-book", "hits": hits("1", "1.1")}],
        )

        assert result.recall is not None
        assert result.recall.mrr == 0.0
        assert result.recall.judged == 1

    def test_the_best_of_several_expectations_wins(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """`expects` lists every node that genuinely answers; best rank wins."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {
                    "q": "grommets",
                    "kind": "mixed",
                    "expects": [{"loc": "1.1"}, {"loc": "1.2"}],
                    "why": "W.",
                }
            ],
            results=[
                {"q": "grommets", "book": "placeholder-book", "hits": hits("9", "1.2", "1.1")}
            ],
        )

        assert result.recall is not None
        assert result.recall.mrr == pytest.approx(0.5)

    def test_a_hit_in_the_wrong_book_does_not_count(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[
                {
                    "q": "grommets",
                    "book": "placeholder-book",
                    "hits": hits("1.2", book="another-book"),
                }
            ],
        )

        assert result.recall is not None
        assert result.recall.mrr == 0.0

    def test_per_kind_slices_are_reported_separately(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Per-kind means are diagnostics — the whole reason the taxonomy exists."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."},
                {
                    "q": "small toothed parts that mesh",
                    "kind": "conceptual",
                    "expects": [{"loc": "1.1"}],
                    "why": "W.",
                },
            ],
            results=[
                {"q": "grommets", "book": "placeholder-book", "hits": hits("1.2")},
                {
                    "q": "small toothed parts that mesh",
                    "book": "placeholder-book",
                    "hits": hits("9", "9.9"),
                },
            ],
        )

        assert result.recall is not None
        assert result.recall.per_kind["exact-term"].mrr == 1.0
        assert result.recall.per_kind["conceptual"].mrr == 0.0

    def test_a_query_the_run_never_asked_is_reported_not_scored(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Silently scoring it zero would make a truncated run look like a
        retrieval collapse."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[],
        )

        assert result.recall is None or result.recall.judged == 0
        assert any("grommets" in note for note in result.unasked)


class TestResultMatching:
    """The documented run-file shape carries `q` and `hits` and nothing else;
    the scope keys are an optional refinement, not a requirement."""

    def test_a_result_without_a_scope_still_matches_its_query(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[{"q": "grommets", "hits": hits("1.2")}],
        )

        assert result.recall is not None
        assert result.recall.mrr == 1.0
        assert not result.unasked

    def test_an_explicit_scope_is_preferred_over_a_scopeless_result(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[
                {"q": "grommets", "hits": hits("9")},
                {"q": "grommets", "book": "placeholder-book", "hits": hits("1.2")},
            ],
        )

        assert result.recall is not None
        assert result.recall.mrr == 1.0


class TestPlaceholderExpectations:
    def test_a_placeholder_expectation_is_reported_as_such_not_as_unasked(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Truth for an area is authored before every book in it exists. Calling
        that "never asked" blames the run for a gap in the truth."""
        assets = write_truth(
            "a-book",
            area="an-area",
            area_queries=[
                {
                    "q": "a cross-book question",
                    "kind": "conceptual",
                    "expects": [{"book": "not-yet-authored", "loc": "TBD"}],
                    "why": "W.",
                }
            ],
        )
        run = load_run(write_run(results=[{"q": "a cross-book question", "hits": []}]))

        result = score(run, load_truth(assets))

        assert not any("never asked" in note for note in result.unasked)
        assert any("placeholder" in note for note in result.skipped)


class TestPerAreaSlices:
    def test_area_query_sets_are_sliced_separately(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """SPEC calls per-area slices diagnostics — they have to exist to be one."""
        assets = write_truth(
            "a-book",
            area="an-area",
            area_queries=[
                {
                    "q": "a cross-book question",
                    "kind": "conceptual",
                    "expects": [{"book": "a-book", "loc": "1.1"}],
                    "why": "W.",
                }
            ],
        )
        run = load_run(
            write_run(
                results=[
                    {"q": "a cross-book question", "area": "an-area", "hits": hits("1.1")}
                ]
            )
        )

        result = score(run, load_truth(assets))

        assert result.recall is not None
        assert "an-area" in result.recall.per_area


class TestIndexPairs:
    def test_unscored_index_pairs_are_reported_not_dropped(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Truth says section-level pairs are scoreable; until a run asks them,
        silence would read as a slice that passed."""
        assets = write_truth("a-book", index_pairs=[{"term": "widget", "loc": "1.1"}])

        result = score(load_run(write_run()), load_truth(assets))

        assert any("index-pair" in note for note in (*result.skipped, *result.unasked))


class TestPooledGuards:
    def test_a_book_with_no_scoreable_nodes_does_not_pool_a_zero(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Pooling 0.0 for a book that has nothing to detect would force a
        regression verdict out of an empty measurement."""
        assets = write_truth("a-book", toc={"front_matter": [{"id": "p", "title": "Preface"}]})
        run = load_run(
            write_run(structure=[{"book": "a-book", "detected": []}])
        )

        result = score(run, load_truth(assets))

        assert "structure" not in result.pooled


class TestGatedSlices:
    def test_a_gated_query_is_skipped_and_reported(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Never silently dropped: it would look like a query that passed."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {
                    "q": "grommets",
                    "kind": "exact-term",
                    "expects": [{"loc": "1.2"}],
                    "why": "W.",
                    "gated": True,
                }
            ],
            results=[{"q": "grommets", "book": "placeholder-book", "hits": hits("9")}],
        )

        assert result.recall is None or result.recall.judged == 0
        assert any("grommets" in note for note in result.skipped)


class TestStructureFidelity:
    """P/R/F1 over (chapter, section) nodes vs toc.yaml."""

    def test_an_exact_tree_scores_one(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            structure=[
                {
                    "book": "placeholder-book",
                    "detected": [
                        {"title": "First Chapter", "kind": "chapter", "position": 0},
                        {"title": "Widgets And Sprockets", "kind": "section", "position": 0},
                        {"title": "Grommets", "kind": "section", "position": 1},
                    ],
                }
            ],
        )

        assert result.structure is not None
        assert result.structure.f1 == 1.0

    def test_titles_match_after_normalisation(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """A detected heading carrying its number, or different case and
        spacing, is the same node."""
        result = scored(
            write_truth,
            write_run,
            structure=[
                {
                    "book": "placeholder-book",
                    "detected": [
                        {"title": "1. FIRST   chapter", "kind": "chapter", "position": 0},
                        {"title": "1.1 widgets and sprockets", "kind": "section", "position": 0},
                        {"title": "1.2  Grommets ", "kind": "section", "position": 1},
                    ],
                }
            ],
        )

        assert result.structure is not None
        assert result.structure.f1 == 1.0

    def test_a_missed_node_costs_recall_and_a_spurious_one_costs_precision(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            structure=[
                {
                    "book": "placeholder-book",
                    "detected": [
                        {"title": "First Chapter", "kind": "chapter", "position": 0},
                        {"title": "Widgets And Sprockets", "kind": "section", "position": 0},
                        {"title": "A Heading The Book Does Not Have", "kind": "section", "position": 1},
                    ],
                }
            ],
        )

        assert result.structure is not None
        assert result.structure.recall == pytest.approx(2 / 3)
        assert result.structure.precision == pytest.approx(2 / 3)

    def test_front_and_back_matter_are_not_scored(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """The dimension is defined over (chapter, section) nodes; a detector
        that skips the index should not be marked down for it."""
        assets = write_truth(
            "placeholder-book",
            toc={
                "front_matter": [{"id": "preface", "title": "Preface"}],
                "chapters": [{"id": "1", "title": "First Chapter"}],
                "back_matter": [{"id": "index", "title": "Index"}],
            },
        )
        run = load_run(
            write_run(
                structure=[
                    {
                        "book": "placeholder-book",
                        "detected": [{"title": "First Chapter", "kind": "chapter", "position": 0}],
                    }
                ]
            )
        )

        result = score(run, load_truth(assets))

        assert result.structure is not None
        assert result.structure.f1 == 1.0


class TestCoverage:
    def test_recall_against_the_concept_inventory(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "placeholder-book",
            concepts=[
                {"concept": "Widget theory", "loc": "1.1"},
                {"concept": "Grommet dynamics", "loc": "1.2"},
            ],
        )
        run = load_run(
            write_run(coverage=[{"book": "placeholder-book", "surfaced": ["Widget theory"]}])
        )

        result = score(run, load_truth(assets))

        assert result.coverage is not None
        assert (result.coverage.found, result.coverage.total) == (1, 2)

    def test_an_accepted_alias_counts_as_surfaced(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """`accept` exists because a knowledge object may legitimately name the
        idea differently."""
        assets = write_truth(
            "placeholder-book",
            concepts=[
                {"concept": "Widget theory", "loc": "1.1", "accept": ["theory of widgets"]},
                {"concept": "Grommet dynamics", "loc": "1.2"},
            ],
        )
        run = load_run(
            write_run(
                coverage=[{"book": "placeholder-book", "surfaced": ["Theory of Widgets"]}]
            )
        )

        result = score(run, load_truth(assets))

        assert result.coverage is not None
        assert result.coverage.found == 1
        assert result.coverage.total == 2
        assert result.coverage.recall == pytest.approx(0.5)


class TestFaithfulness:
    def test_mean_claim_support_ratio(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            faithfulness=[
                {"book": "placeholder-book", "loc": "1.1", "supported": 9, "claims": 10},
                {"book": "placeholder-book", "loc": "1.2", "supported": 7, "claims": 10},
            ],
        )

        assert result.faithfulness is not None
        assert result.faithfulness.mean == pytest.approx(0.8)

    def test_summaries_below_the_pass_bar_are_named(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        result = scored(
            write_truth,
            write_run,
            faithfulness=[
                {"book": "placeholder-book", "loc": "1.1", "supported": 9, "claims": 10},
                {"book": "placeholder-book", "loc": "1.2", "supported": 5, "claims": 10},
            ],
        )

        assert result.faithfulness is not None
        assert [entry.loc for entry in result.faithfulness.below_bar] == ["1.2"]

    def test_a_summary_with_no_claims_is_skipped_not_scored_as_zero(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Nothing to verify is not the same as nothing supported; averaging it
        in as 0.0 would report a faithfulness collapse."""
        result = scored(
            write_truth,
            write_run,
            faithfulness=[{"book": "placeholder-book", "loc": "1.1", "supported": 0, "claims": 0}],
        )

        assert result.faithfulness is None
        assert any("no claims" in note for note in result.skipped)


class TestCost:
    def test_cost_is_carried_but_never_scored(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Tracked, never scored — a cheaper run is not automatically better."""
        result = scored(
            write_truth,
            write_run,
            cost={"input_tokens": 100, "output_tokens": 50, "wall_seconds": 12.5},
        )

        assert result.cost is not None
        assert result.cost.input_tokens == 100
        assert "cost" not in result.pooled


class TestRegression:
    def make(
        self,
        write_truth: Callable[..., Path],
        write_run: Callable[..., Path],
        *,
        name: str,
        ranks: tuple[str, ...],
    ) -> object:
        assets = write_truth("placeholder-book", queries=[
            {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
        ])
        run = load_run(
            write_run(
                name,
                results=[{"q": "grommets", "book": "placeholder-book", "hits": hits(*ranks)}],
            )
        )
        return score(run, load_truth(assets))

    def test_an_identical_run_is_not_a_regression(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base = self.make(write_truth, write_run, name="a.json", ranks=("1.2",))
        cand = self.make(write_truth, write_run, name="b.json", ranks=("1.2",))

        assert not compare(base, cand).is_regression

    def test_a_drop_beyond_the_threshold_is_a_regression(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base = self.make(write_truth, write_run, name="a.json", ranks=("1.2",))
        cand = self.make(write_truth, write_run, name="b.json", ranks=("9", "9.9", "1.2"))

        verdict = compare(base, cand)

        assert verdict.is_regression
        assert any(change.dimension == "recall" for change in verdict.regressions)

    def test_a_drop_inside_the_threshold_is_not(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """MRR tolerance is 0.05. Twenty queries, one slipping from rank 1 to
        rank 2, moves the pooled mean by 0.025 — noise, not a regression."""
        queries = [
            {"q": f"term {n}", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            for n in range(20)
        ]
        assets = write_truth("placeholder-book", queries=queries)
        truth = load_truth(assets)

        def run_with(slipped: int) -> object:
            results = [
                {
                    "q": f"term {n}",
                    "book": "placeholder-book",
                    "hits": hits("9", "1.2") if n < slipped else hits("1.2"),
                }
                for n in range(20)
            ]
            return score(load_run(write_run(f"r{slipped}.json", results=results)), truth)

        verdict = compare(run_with(0), run_with(1))

        assert not verdict.is_regression

    def test_an_improvement_is_never_a_regression(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base = self.make(write_truth, write_run, name="a.json", ranks=("9", "1.2"))
        cand = self.make(write_truth, write_run, name="b.json", ranks=("1.2",))

        assert not compare(base, cand).is_regression

    def test_thresholds_cover_every_scored_dimension(self) -> None:
        assert set(THRESHOLDS) == {"recall", "structure", "coverage", "faithfulness"}


class TestPurity:
    def test_the_scorer_imports_nothing_from_the_pipeline(self) -> None:
        """`score` compares run file x truth "with zero pipeline imports". A
        scorer that could import the pipeline could accidentally measure it."""
        source = (
            Path(__file__).parent.parent
            / "src"
            / "booksmart_bench"
            / "scoring.py"
        ).read_text()

        assert "booksmart_core" not in source

    def test_scoring_module_does_not_pull_core_in_transitively(self) -> None:
        import subprocess
        import sys

        probe = (
            "import sys; import booksmart_bench.scoring; "
            "print(any(m == 'booksmart_core' or m.startswith('booksmart_core.') "
            "for m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )

        assert out.stdout.strip() == "False", out.stdout


class TestSerialisation:
    def test_scores_round_trip_as_json(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """`report` renders from scores; scores have to survive being written."""
        result = scored(
            write_truth,
            write_run,
            queries=[
                {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."}
            ],
            results=[{"q": "grommets", "book": "placeholder-book", "hits": hits("1.2")}],
        )

        assert json.loads(json.dumps(result.as_dict()))["pooled"]["recall"] == 1.0
