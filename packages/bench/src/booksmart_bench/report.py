"""Two runs, side by side, as Markdown.

The house style is the research write-ups in ``docs/research/``: a stated answer
first, then generated tables, then the diagnostics that explain where the
difference came from. The tables are generated and the conclusion is written —
this module produces the former and leaves room for the latter.

Two things it refuses to do. It never blends dimensions into a headline number,
because an MRR and an F1 do not average into anything. And it never lets a
per-slice figure read as a verdict: slices are labelled diagnostics, and only
the pooled aggregate carries the regression call.

Like the scorer it renders, this imports nothing from the pipeline.
"""

from collections.abc import Iterable, Sequence

from booksmart_bench.scoring import THRESHOLDS, Cost, RecallScore, Scores, Verdict

# Ordered so the page reads from the dimension a reader most likely came for.
DIMENSION_ORDER = ("recall", "structure", "coverage", "faithfulness")


def render(baseline: Scores, candidate: Scores, verdict: Verdict) -> str:
    """The full comparison page."""
    lines: list[str] = []
    lines += _heading(baseline, candidate)
    lines += _answer(verdict)
    lines += _pooled_table(baseline, candidate, verdict)
    lines += _diagnostics(baseline, candidate)
    lines += _cost(baseline, candidate)
    lines += _not_measured(baseline, candidate)
    return "\n".join(lines).rstrip() + "\n"


def _heading(baseline: Scores, candidate: Scores) -> list[str]:
    return [
        "# Benchmark comparison",
        "",
        f"- **baseline** `{baseline.run_id}` — snapshot `{baseline.snapshot}`",
        f"- **candidate** `{candidate.run_id}` — snapshot `{candidate.snapshot}`",
        "",
        "A score is only comparable against a run of the same shape; the snapshots"
        " above are what that means in practice.",
        "",
    ]


def _answer(verdict: Verdict) -> list[str]:
    lines = ["## Answer", ""]
    if verdict.is_regression:
        named = ", ".join(f"`{change.dimension}`" for change in verdict.regressions)
        lines += [
            f"**REGRESSION** — {named} fell further than the per-dimension threshold allows.",
            "",
        ]
        for change in verdict.regressions:
            lines.append(
                f"- `{change.dimension}`: {change.baseline:.3f} -> {change.candidate:.3f} "
                f"({_signed(change.delta)}, tolerance {THRESHOLDS[change.dimension]:.2f})"
            )
        lines.append("")
    else:
        lines += [
            "**No regression.** Every pooled dimension held within its threshold.",
            "",
        ]
    if verdict.incomparable:
        named = ", ".join(f"`{name}`" for name in verdict.incomparable)
        lines += [
            f"> One run scored {named} and the other did not, so those dimensions are"
            " **not comparable** between these two runs.",
            "",
        ]
    return lines


def _pooled_table(baseline: Scores, candidate: Scores, verdict: Verdict) -> list[str]:
    changes = {change.dimension: change for change in verdict.changes}
    base_pooled, cand_pooled = baseline.pooled, candidate.pooled

    lines = [
        "## Pooled",
        "",
        "The only figures a pipeline variant is judged on.",
        "",
        "| dimension | baseline | candidate | delta | tolerance | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dimension in DIMENSION_ORDER:
        in_base, in_cand = dimension in base_pooled, dimension in cand_pooled
        if not (in_base or in_cand):
            continue
        change = changes.get(dimension)
        if change is None:
            lines.append(
                f"| {dimension} | {_cell(base_pooled.get(dimension))} | "
                f"{_cell(cand_pooled.get(dimension))} | — | "
                f"{THRESHOLDS[dimension]:.2f} | not comparable |"
            )
            continue
        regressed = change in verdict.regressions
        lines.append(
            f"| {dimension} | {change.baseline:.3f} | {change.candidate:.3f} | "
            f"{_signed(change.delta)} | {THRESHOLDS[dimension]:.2f} | "
            f"{'**REGRESSION**' if regressed else 'ok'} |"
        )
    lines.append("")
    return lines


def _diagnostics(baseline: Scores, candidate: Scores) -> list[str]:
    if baseline.recall is None and candidate.recall is None:
        return []

    lines = [
        "## Diagnostics",
        "",
        "Per-slice figures. These say *where* a difference came from and decide"
        " nothing — only the pooled aggregate above compares variants.",
        "",
    ]

    kinds = sorted(
        {*_kinds(baseline), *_kinds(candidate)},
    )
    if kinds:
        lines += [
            "### Recall by query kind",
            "",
            "| kind | baseline MRR | candidate MRR | baseline hit@5 | candidate hit@5 | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for kind in kinds:
            base_slice = _kind_slice(baseline, kind)
            cand_slice = _kind_slice(candidate, kind)
            judged = (cand_slice or base_slice)
            lines.append(
                f"| {kind} | {_cell(base_slice.mrr if base_slice else None)} | "
                f"{_cell(cand_slice.mrr if cand_slice else None)} | "
                f"{_cell(base_slice.hit_at_5 if base_slice else None)} | "
                f"{_cell(cand_slice.hit_at_5 if cand_slice else None)} | "
                f"{judged.judged if judged else 0} |"
            )
        lines.append("")

    attributions = [
        s.recall.attribution
        for s in (baseline, candidate)
        if s.recall is not None and s.recall.attribution is not None
    ]
    if attributions:
        lines += [
            "Cross-book attribution (right book anywhere in the top k): "
            + " -> ".join(f"{value:.3f}" for value in attributions),
            "",
        ]

    below = candidate.faithfulness.below_bar if candidate.faithfulness else ()
    if below:
        lines += ["### Summaries below the faithfulness bar", ""]
        for entry in below:
            ratio = entry.ratio or 0.0
            lines.append(f"- `{entry.book}` {entry.loc} — {ratio:.2f}")
        lines.append("")
    return lines


def _cost(baseline: Scores, candidate: Scores) -> list[str]:
    if baseline.cost is None and candidate.cost is None:
        return []
    lines = [
        "## Cost and throughput",
        "",
        "Tracked, **never scored** — a cheaper run is not automatically a better one.",
        "",
        "| measure | baseline | candidate |",
        "| --- | --- | --- |",
    ]
    rows = (
        ("input tokens", "input_tokens"),
        ("output tokens", "output_tokens"),
        ("embedding tokens", "embedding_tokens"),
        ("wall seconds", "wall_seconds"),
    )
    for label, attribute in rows:
        lines.append(
            f"| {label} | {_raw(baseline.cost, attribute)} | {_raw(candidate.cost, attribute)} |"
        )
    lines.append("")
    return lines


def _not_measured(baseline: Scores, candidate: Scores) -> list[str]:
    skipped = _unique(baseline.skipped, candidate.skipped)
    unasked = _unique(baseline.unasked, candidate.unasked)
    if not skipped and not unasked:
        return []

    lines = [
        "## Not measured",
        "",
        "Listed rather than dropped: a slice that disappears from a report is a"
        " slice that reads as having passed.",
        "",
    ]
    if skipped:
        lines += ["**Skipped**", ""] + [f"- {note}" for note in skipped] + [""]
    if unasked:
        lines += ["**In truth, never asked by these runs**", ""]
        lines += [f"- {note}" for note in unasked] + [""]
    return lines


# --- helpers ----------------------------------------------------------------


def _kinds(scores: Scores) -> Iterable[str]:
    return () if scores.recall is None else scores.recall.per_kind.keys()


def _kind_slice(scores: Scores, kind: str) -> RecallScore | None:
    if scores.recall is None:
        return None
    return scores.recall.per_kind.get(kind)


def _cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _raw(cost: Cost | None, attribute: str) -> str:
    if cost is None:
        return "—"
    value = getattr(cost, attribute)
    return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"


def _signed(delta: float) -> str:
    return f"{delta:+.3f}"


def _unique(*groups: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for group in groups:
        for note in group:
            seen.setdefault(note, None)
    return list(seen)
