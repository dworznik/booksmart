"""Two runs rendered side by side, in the house style of the research write-ups."""

from collections.abc import Callable
from pathlib import Path

from booksmart_bench.report import render
from booksmart_bench.scoring import compare, load_run, score
from booksmart_bench.truth import load_truth

from .conftest import hits

QUERIES = [
    {"q": "grommets", "kind": "exact-term", "expects": [{"loc": "1.2"}], "why": "W."},
    {
        "q": "small toothed parts that mesh",
        "kind": "conceptual",
        "expects": [{"loc": "1.1"}],
        "why": "W.",
    },
]


def two_runs(
    write_truth: Callable[..., Path],
    write_run: Callable[..., Path],
    *,
    baseline_ranks: tuple[str, ...],
    candidate_ranks: tuple[str, ...],
    **sections: object,
) -> tuple[object, object]:
    truth = load_truth(write_truth("placeholder-book", queries=QUERIES))

    def build(name: str, ranks: tuple[str, ...]) -> object:
        results = [
            {"q": "grommets", "book": "placeholder-book", "hits": hits(*ranks)},
            {
                "q": "small toothed parts that mesh",
                "book": "placeholder-book",
                "hits": hits("1.1"),
            },
        ]
        return score(load_run(write_run(name, results=results, **sections)), truth)

    return build("base.json", baseline_ranks), build("cand.json", candidate_ranks)


class TestRender:
    def test_puts_both_runs_in_one_table(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth, write_run, baseline_ranks=("1.2",), candidate_ranks=("1.2",)
        )

        markdown = render(base, cand, compare(base, cand))

        assert "| dimension |" in markdown
        assert "recall" in markdown

    def test_names_both_runs_and_their_snapshots(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """A score is only comparable against a run of the same shape, so the
        snapshots have to be legible on the page."""
        base, cand = two_runs(
            write_truth, write_run, baseline_ranks=("1.2",), candidate_ranks=("1.2",)
        )

        markdown = render(base, cand, compare(base, cand))

        assert base.run_id in markdown and base.snapshot in markdown

    def test_a_regression_is_stated_not_left_to_the_reader(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth,
            write_run,
            baseline_ranks=("1.2",),
            candidate_ranks=("9", "8", "1.2"),
        )

        markdown = render(base, cand, compare(base, cand))

        assert "REGRESSION" in markdown

    def test_no_regression_says_so(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth, write_run, baseline_ranks=("1.2",), candidate_ranks=("1.2",)
        )

        markdown = render(base, cand, compare(base, cand))

        assert "No regression" in markdown

    def test_deltas_carry_a_sign(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth,
            write_run,
            baseline_ranks=("9", "1.2"),
            candidate_ranks=("1.2",),
        )

        markdown = render(base, cand, compare(base, cand))

        assert "+0." in markdown

    def test_per_kind_slices_are_rendered_as_diagnostics(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth, write_run, baseline_ranks=("1.2",), candidate_ranks=("1.2",)
        )

        markdown = render(base, cand, compare(base, cand))

        assert "conceptual" in markdown
        assert "exact-term" in markdown

    def test_diagnostics_are_marked_as_not_deciding_anything(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Only the pooled aggregate compares variants; the report must not let
        a per-slice number read as a verdict."""
        base, cand = two_runs(
            write_truth, write_run, baseline_ranks=("1.2",), candidate_ranks=("1.2",)
        )

        markdown = render(base, cand, compare(base, cand))

        assert "diagnostic" in markdown.lower()

    def test_cost_is_shown_but_marked_unscored(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        base, cand = two_runs(
            write_truth,
            write_run,
            baseline_ranks=("1.2",),
            candidate_ranks=("1.2",),
            cost={"input_tokens": 1000, "output_tokens": 200, "wall_seconds": 30.0},
        )

        markdown = render(base, cand, compare(base, cand))

        assert "1000" in markdown or "1,000" in markdown
        assert "never scored" in markdown.lower()

    def test_skipped_and_unasked_slices_reach_the_page(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """The whole reason they are collected: a gated slice that vanished
        from the report is a gated slice that looks like it passed."""
        truth = load_truth(
            write_truth(
                "placeholder-book",
                queries=[
                    {
                        "q": "grommets",
                        "kind": "exact-term",
                        "expects": [{"loc": "1.2"}],
                        "why": "W.",
                        "gated": True,
                    },
                    {
                        "q": "never asked",
                        "kind": "conceptual",
                        "expects": [{"loc": "1.1"}],
                        "why": "W.",
                    },
                ],
            )
        )
        both = [score(load_run(write_run(f"{n}.json", results=[])), truth) for n in "ab"]

        markdown = render(both[0], both[1], compare(both[0], both[1]))

        assert "gated" in markdown.lower()
        assert "never asked" in markdown

    def test_a_dimension_only_one_run_scored_is_flagged(
        self, write_truth: Callable[..., Path], write_run: Callable[..., Path]
    ) -> None:
        """Two runs measuring different dimensions are not comparable, and the
        page has to say so rather than show a blank cell."""
        truth = load_truth(
            write_truth(
                "placeholder-book",
                queries=QUERIES,
                concepts=[{"concept": "Widget theory", "loc": "1.1"}],
            )
        )
        results = [{"q": "grommets", "book": "placeholder-book", "hits": hits("1.2")}]
        base = score(load_run(write_run("a.json", results=results)), truth)
        cand = score(
            load_run(
                write_run(
                    "b.json",
                    results=results,
                    coverage=[{"book": "placeholder-book", "surfaced": ["Widget theory"]}],
                )
            ),
            truth,
        )

        markdown = render(base, cand, compare(base, cand))

        assert "not comparable" in markdown.lower()
