"""The run verb: the real search path in, a location-keyed run file out.

The corpus these tests measure is real — a two-page PDF taken through the same
public Stage functions the product runs, against core's deterministic stand-in
providers. What is deliberately *not* real is the ranking: fake embeddings are
derived from text length, so which record comes back first is arbitrary. These
tests are therefore about the artefact — that queries were asked through
``search()``, that every hit carries a ToC node id rather than a record id, and
that what a run cannot measure it names — never about a score being good.
"""

import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from booksmart_bench.config import load_settings
from booksmart_bench.errors import BenchError
from booksmart_bench.execute import (
    RunOutcome,
    _Drops,
    _Joined,
    execute_run,
    locate,
)
from booksmart_bench.ingest import ingest_scope
from booksmart_bench.judge import JudgeConfig
from booksmart_bench.locations import Detected, LocationMap
from booksmart_bench.scoring import load_run
from booksmart_bench.scoring import score as score_run
from booksmart_bench.truth import load_truth
from booksmart_core.models import Chapter
from booksmart_core.search import SearchHit, SearchResults

from .conftest import StubLLM  # type: ignore[import-not-found]

pytestmark = pytest.mark.usefixtures("fake_providers")

QUERIES: list[dict[str, object]] = [
    {
        "q": "grommets",
        "kind": "exact-term",
        "expects": [{"loc": "1.2"}],
        "why": "The book's own term for the part.",
    },
    {
        "q": "what keeps a joint from leaking",
        "kind": "conceptual",
        "expects": [{"loc": "1.2"}],
        "why": "Asks for the idea without the word.",
    },
]

TOC: dict[str, object] = {
    "chapters": [
        {
            "id": "1",
            "title": "First Chapter",
            "sections": [
                {"id": "1.1", "title": "Widgets And Sprockets"},
                {"id": "1.2", "title": "Grommets"},
            ],
        },
        {"id": "2", "title": "Second Chapter"},
    ]
}


@pytest.fixture()
def corpus_assets(
    write_truth: Callable[..., Path], place_source: Callable[..., str]
) -> Path:
    """An assets checkout whose one book has been ingested into the corpus."""
    root = write_truth(
        "a-book",
        book={"title": "A Book", "area": "an-area"},
        toc=TOC,
        queries=QUERIES,
        concepts=[
            {"concept": "Fake determinism", "loc": "1.1"},
            {"concept": "Something never extracted", "loc": "1.2"},
        ],
    )
    place_source(root, "a-book")
    ingest_scope(root, load_truth(root), "a-book", load_settings())
    return root


def run(assets: Path, family: str = "all", scope: str = "a-book", **kwargs: Any) -> RunOutcome:
    return execute_run(assets, load_truth(assets), family, scope, load_settings(), **kwargs)


def document(outcome: RunOutcome) -> dict[str, object]:
    return json.loads(outcome.path.read_text())


class TestTheArtefact:
    def test_the_run_file_lands_in_runs(self, corpus_assets: Path) -> None:
        outcome = run(corpus_assets)

        assert outcome.path.parent == corpus_assets / "runs"
        assert outcome.path.name.endswith(".json")

    def test_the_run_id_names_the_family_and_the_scope(self, corpus_assets: Path) -> None:
        outcome = run(corpus_assets, "recall")

        assert outcome.run_id.endswith("-recall-a-book")

    def test_out_overrides_where_it_is_written(
        self, corpus_assets: Path, tmp_path: Path
    ) -> None:
        outcome = run(corpus_assets, out=tmp_path / "elsewhere.json")

        assert outcome.path == tmp_path / "elsewhere.json"
        assert outcome.path.is_file()

    def test_the_configuration_is_stamped(self, corpus_assets: Path) -> None:
        """A score is only comparable against a run of the same shape, so the
        shape travels with the run rather than living in someone's shell
        history."""
        stamped = document(run(corpus_assets))["corpus"]

        assert isinstance(stamped, dict)
        assert stamped["snapshot"]
        assert stamped["books"] == ["a-book"]
        # The configuration the snapshot is computed from, spelled out: the
        # directory name says two runs differ, this says how.
        assert stamped["config"]["embedding_provider"] == "fake"
        assert stamped["config"]["summary_prompt_version"]

    def test_the_search_parameters_are_stamped(self, corpus_assets: Path) -> None:
        search = document(run(corpus_assets, "recall", limit=3))["search"]

        assert isinstance(search, dict)
        assert search["limit"] == 3
        assert search["mode"]
        assert search["record_types"] == ["chapter", "section", "knowledge_object"]


class TestRecall:
    def test_every_query_truth_holds_is_asked(self, corpus_assets: Path) -> None:
        results = document(run(corpus_assets, "recall"))["results"]

        assert isinstance(results, list)
        assert sorted(str(result["q"]) for result in results) == sorted(
            str(query["q"]) for query in QUERIES
        )

    def test_hits_carry_locations_not_record_ids(self, corpus_assets: Path) -> None:
        """The join the run writer owns: a scorer that had to resolve a record
        id would need the corpus, which is the coupling the artefact split
        exists to prevent."""
        results = document(run(corpus_assets, "recall"))["results"]

        assert isinstance(results, list)
        located = [hit for result in results for hit in result["hits"]]
        assert located, "the corpus returned no hits at all"
        assert {hit["loc"] for hit in located} <= {"1", "1.1", "1.2", "2"}
        assert all(hit["book"] == "a-book" for hit in located)

    def test_ranks_are_the_ranks_search_gave(self, corpus_assets: Path) -> None:
        results = document(run(corpus_assets, "recall"))["results"]

        assert isinstance(results, list)
        for result in results:
            ranks = [hit["rank"] for hit in result["hits"]]
            assert ranks == sorted(ranks)
            assert len(set(ranks)) == len(ranks)

    def test_a_book_query_is_answered_from_its_own_book(self, corpus_assets: Path) -> None:
        results = document(run(corpus_assets, "recall"))["results"]

        assert isinstance(results, list)
        assert all(result["book"] == "a-book" for result in results)

    def test_a_gated_query_is_skipped_and_named(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        """Gated truth waits on a pipeline feature; asking it would spend an
        embedding call to measure a miss it could never have hit."""
        root = write_truth(
            "a-book",
            book={"title": "A Book", "area": "an-area"},
            toc=TOC,
            queries=[
                {
                    "q": "a passage-level question",
                    "kind": "mixed",
                    "expects": [{"loc": "1.2"}],
                    "why": "Needs passage records.",
                    "gated": True,
                }
            ],
        )
        place_source(root, "a-book")
        ingest_scope(root, load_truth(root), "a-book", load_settings())

        outcome = run(root, "recall")

        assert document(outcome)["results"] == []
        assert any("gated" in note for note in outcome.notes)

    def test_an_area_scope_asks_the_areas_cross_book_queries(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        root = write_truth(
            "a-book",
            book={"title": "A Book", "area": "an-area"},
            toc=TOC,
            area="an-area",
            area_queries=[
                {
                    "q": "which book covers grommets",
                    "kind": "mixed",
                    "expects": [{"book": "a-book", "loc": "1.2"}],
                    "why": "Attribution across the area's books.",
                }
            ],
        )
        place_source(root, "a-book")
        ingest_scope(root, load_truth(root), "a-book", load_settings())

        results = document(run(root, "recall", "an-area"))["results"]

        assert isinstance(results, list)
        assert [result["area"] for result in results] == ["an-area"]


class TestIngestionFamily:
    def test_detected_structure_is_reported_in_document_order(
        self, corpus_assets: Path
    ) -> None:
        structure = document(run(corpus_assets, "ingestion"))["structure"]

        assert isinstance(structure, list)
        detected = structure[0]["detected"]
        assert [node["title"] for node in detected] == [
            "First Chapter",
            "Widgets And Sprockets",
            "Grommets",
            "Second Chapter",
        ]
        assert [node["position"] for node in detected] == [0, 1, 2, 3]

    def test_extraction_coverage_reports_what_was_surfaced(
        self, corpus_assets: Path
    ) -> None:
        coverage = document(run(corpus_assets, "ingestion"))["coverage"]

        assert isinstance(coverage, list)
        assert coverage[0]["book"] == "a-book"
        # The fake extractor surfaces one knowledge object per chapter — and
        # nothing at all for the second authored concept, which is the half of
        # the coverage dimension that can actually fall.
        assert "Fake determinism" in coverage[0]["surfaced"]
        assert "Something never extracted" not in coverage[0]["surfaced"]

    def test_the_ingestion_families_cost_comes_from_the_corpus_provenance(
        self, corpus_assets: Path
    ) -> None:
        """Cost is per book per Stage, and the Stage breakdown only exists in
        the provenance the ingest wrote (booksmart#65)."""
        cost = document(run(corpus_assets, "ingestion"))["cost"]

        assert isinstance(cost, dict)
        stages = {str(entry["stage"]) for entry in cost["per_stage"]}
        assert {"parse", "structure", "extraction", "summaries", "embeddings"} <= stages

    def test_a_recall_run_costs_only_what_it_spent(self, corpus_assets: Path) -> None:
        """A recall run did not ingest anything; carrying the ingest's spend
        would double-count it against the run that really paid."""
        cost = document(run(corpus_assets, "recall"))["cost"]

        assert isinstance(cost, dict)
        assert {str(entry["stage"]) for entry in cost["per_stage"]} == {"search"}


class TestJudging:
    def test_faithfulness_is_judged_against_the_source_slice(
        self, corpus_assets: Path
    ) -> None:
        judge = StubLLM(claims='["a claim"]')

        outcome = run(corpus_assets, "ingestion", judge_llm=judge, judge=_judge_config())

        faithfulness = document(outcome)["faithfulness"]
        assert isinstance(faithfulness, list)
        assert faithfulness
        assert all(entry["loc"] in {"1", "1.1", "1.2", "2"} for entry in faithfulness)
        assert any("Grommets seal the joint" in prompt for prompt in judge.prompts)

    def test_the_judge_identity_is_stamped(self, corpus_assets: Path) -> None:
        outcome = run(
            corpus_assets, "ingestion", judge_llm=StubLLM(), judge=_judge_config()
        )

        stamped = document(outcome)["judge"]
        assert isinstance(stamped, dict)
        assert (stamped["provider"], stamped["model"]) == ("stub", "a-judge-model")
        assert stamped["prompt_version"]

    def test_judge_spend_lands_in_the_cost_dimension(self, corpus_assets: Path) -> None:
        outcome = run(
            corpus_assets, "ingestion", judge_llm=StubLLM(), judge=_judge_config()
        )

        cost = document(outcome)["cost"]
        assert isinstance(cost, dict)
        assert any(str(entry["stage"]) == "judge" for entry in cost["per_stage"])

    def test_a_recall_run_does_not_stamp_a_judge_it_never_used(
        self, corpus_assets: Path
    ) -> None:
        """A judge is configured for the whole invocation, but only the
        ingestion family picks it up; naming it anyway would describe a
        faithfulness measurement that never happened."""
        outcome = run(
            corpus_assets, "recall", judge_llm=StubLLM(), judge=_judge_config()
        )

        assert document(outcome)["judge"] is None

    def test_without_a_judge_the_dimension_is_absent_and_named(
        self, corpus_assets: Path
    ) -> None:
        outcome = run(corpus_assets, "ingestion")

        assert document(outcome).get("faithfulness") == []
        assert any("judge" in note for note in outcome.notes)


class TestNothingSilentlyUnasked:
    def test_a_book_with_no_queries_says_so(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        """Truth's own gap, and invisible from the results alone: a book nothing
        was asked about scores nothing."""
        root = write_truth(
            "a-book", book={"title": "A Book", "area": "an-area"}, toc=TOC, queries=[]
        )
        place_source(root, "a-book")
        ingest_scope(root, load_truth(root), "a-book", load_settings())

        outcome = run(root, "recall")

        assert any("no queries authored" in note for note in outcome.notes)

    def test_an_area_with_no_query_file_says_so(self, corpus_assets: Path) -> None:
        """The books all name this area, but nobody authored its cross-book
        set — so the area slice is missing, not empty."""
        outcome = run(corpus_assets, "recall", "an-area")

        assert any("no cross-book queries were asked" in note for note in outcome.notes)

    def test_an_area_whose_query_file_is_empty_says_so(
        self, write_truth: Callable[..., Path], place_source: Callable[..., str]
    ) -> None:
        root = write_truth(
            "a-book",
            book={"title": "A Book", "area": "an-area"},
            toc=TOC,
            area="an-area",
            area_queries=[],
        )
        place_source(root, "a-book")
        ingest_scope(root, load_truth(root), "a-book", load_settings())

        outcome = run(root, "recall", "an-area")

        assert any("no cross-book queries authored" in note for note in outcome.notes)


class TestWhatItRefuses:
    def test_an_unknown_family_names_the_families(self, corpus_assets: Path) -> None:
        with pytest.raises(BenchError, match="recall"):
            run(corpus_assets, "everything")

    def test_a_book_that_was_never_ingested_stops_the_run(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A run over an empty corpus would emit a run file of zeroes, which
        scores exactly like a pipeline that lost every record."""
        root = write_truth("a-book", book={"title": "A Book", "area": "an-area"}, toc=TOC)

        with pytest.raises(BenchError, match="ingest"):
            run(root)


class TestTheJoin:
    """The contract the whole artefact split rests on, exercised directly: the
    fixture corpus resolves every hit, so a corpus-level test could not fail for
    the reason a dropped hit exists."""

    def test_a_dropped_hit_leaves_the_surviving_ranks_alone(self) -> None:
        """Renumbering around a drop would promote a hit the pipeline did not —
        and MRR is exactly the rank of the first right answer."""
        book_id = uuid.uuid4()
        maps = {
            "a-book": _Joined(
                book_id=book_id,
                nodes=(),
                locations=LocationMap(by_record={"r1": "1.1", "r3": "1.2"}),
            )
        }

        located = locate(
            SearchResults(
                hits=[_hit("r1", book_id), _hit("r2", book_id), _hit("r3", book_id)],
                embedding_tokens=0,
            ),
            maps,
            {book_id: "a-book"},
            _Drops(),
        )

        assert [hit["rank"] for hit in located] == [1, 3]
        assert [hit["loc"] for hit in located] == ["1.1", "1.2"]

    def test_every_drop_is_counted(self) -> None:
        book_id, elsewhere = uuid.uuid4(), uuid.uuid4()
        maps = {"a-book": _Joined(book_id=book_id, nodes=(), locations=LocationMap())}
        drops = _Drops()

        locate(
            SearchResults(hits=[_hit("r1", book_id), _hit("r2", elsewhere)], embedding_tokens=0),
            maps,
            {book_id: "a-book"},
            drops,
        )

        assert drops.unjoined["a-book"] == 1
        assert drops.unknown_book == 1
        assert len(drops.notes()) == 2


def _hit(record_id: str, book_id: uuid.UUID) -> SearchHit:
    """A search hit over a chapter row, with only what the join reads."""
    return SearchHit(
        score=1.0,
        record_type="chapter",
        record_id=record_id,  # type: ignore[arg-type]
        book_id=book_id,
        text="",
        row=Chapter(title="A Chapter"),
    )


class TestRoundTrip:
    def test_the_emitted_run_file_scores(self, corpus_assets: Path) -> None:
        """The artefacts have to meet: whatever `run` writes, `score` reads."""
        outcome = run(corpus_assets)

        scores = score_run(load_run(outcome.path), load_truth(corpus_assets))

        assert scores.recall is not None and scores.recall.judged == len(QUERIES)
        assert scores.structure is not None and scores.structure.expected == 4
        assert scores.coverage is not None and scores.coverage.total == 2

    def test_a_judged_run_scores_its_faithfulness(self, corpus_assets: Path) -> None:
        outcome = run(
            corpus_assets,
            judge_llm=StubLLM(claims='["a claim"]', supported=True),
            judge=_judge_config(),
        )

        scores = score_run(load_run(outcome.path), load_truth(corpus_assets))

        assert scores.faithfulness is not None
        assert scores.faithfulness.mean == 1.0

    def test_no_record_id_reaches_the_run_file(self, corpus_assets: Path) -> None:
        """The invariant the artefact split rests on. Checked over the whole
        measured document rather than the hits alone, so a dimension added later
        cannot leak one quietly."""
        measured = document(run(corpus_assets))
        payload = json.dumps(
            {key: measured[key] for key in ("results", "structure", "coverage", "faithfulness")}
        )

        assert not re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", payload
        )

    def test_a_summary_the_judge_could_not_read_is_reported_by_the_scorer(
        self, corpus_assets: Path
    ) -> None:
        """The reason travels from the judge into the run file into the report —
        "no claims to verify" and "the judge broke" are different stories."""
        outcome = run(
            corpus_assets, judge_llm=StubLLM(claims="not json"), judge=_judge_config()
        )

        scores = score_run(load_run(outcome.path), load_truth(corpus_assets))

        assert any("judge" in note for note in scores.skipped)


def _judge_config() -> JudgeConfig:
    return JudgeConfig(provider="stub", model="a-judge-model")
