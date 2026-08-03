"""Fixture corpus and harness for the hybrid-vs-dense retrieval eval (issue #40).

Not a test module (no ``test_`` prefix, so pytest does not collect it): this is
the machinery ``test_hybrid_eval.py`` drives. It answers the one question the
hybrid-search work could not settle from primary sources — does fusing BM25 with
dense retrieval actually beat dense alone on booksmart's own shape of corpus,
which is chapter/section summaries and knowledge objects rather than raw prose?

The corpus stands in for *A Philosophy of Software Design*, the book the test
suite has always used as its example. It is written by hand rather than ingested
from the PDF, deliberately: an eval needs to know which record *should* answer
each query, and that ground truth has to be authored, not inferred. Records are
sized and shaped like the ones the summaries and extraction Stages emit.

Queries come in three kinds, and the split is the whole point of the exercise:

- ``exact-term`` — the book's own vocabulary. The term appears verbatim in the
  expected records and nowhere else. Dense retrieval has no special claim on a
  coined phrase, so this is where sparse should earn its place.
- ``conceptual`` — a paraphrase that shares no content word with the record it
  should find. Sparse retrieval can do nothing here by construction, so this is
  where fusion must be shown not to *hurt*.
- ``mixed`` — a real question: some shared vocabulary, some paraphrase. The
  ordinary case, and the one a default should be chosen on.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from booksmart_core.llm import EmbeddingProvider
from booksmart_core.models import Chapter, KnowledgeObject, Section
from booksmart_core.search import SearchMode, search
from booksmart_core.sparse import SparseEmbeddingProvider
from booksmart_core.vectors import RecordType, VectorRecord, VectorStore

QueryKind = Literal["exact-term", "conceptual", "mixed"]


@dataclass(frozen=True)
class FixtureRecord:
    """One embeddable record, with a stable handle queries can name."""

    key: str
    record_type: RecordType
    title: str
    text: str


@dataclass(frozen=True)
class FixtureQuery:
    query: str
    kind: QueryKind
    # Record keys that genuinely answer this query. Order is irrelevant; being
    # in the top few is what is measured.
    expects: tuple[str, ...]
    why: str


# --- the corpus -------------------------------------------------------------
#
# Chapter and section summaries read like the summaries Stage writes (a few
# sentences, no headings); knowledge objects read like the extraction Stage's
# output, whose embedded text is "type: title\nsummary\ncontent".

CORPUS: tuple[FixtureRecord, ...] = (
    FixtureRecord(
        key="ch-complexity",
        record_type="chapter",
        title="The Nature of Complexity",
        text=(
            "Complexity is anything about the structure of a software system that "
            "makes it hard to understand and modify. It is caused by dependencies "
            "and obscurity, and it shows up as change amplification, cognitive "
            "load, and unknown unknowns. Complexity is incremental: it accumulates "
            "from many small decisions rather than arriving all at once."
        ),
    ),
    FixtureRecord(
        key="ko-change-amplification",
        record_type="knowledge_object",
        title="Change amplification",
        text=(
            "Symptom: Change amplification\n"
            "A seemingly simple change requires edits in many different places.\n"
            "Change amplification is the first symptom of complexity. A banner "
            "colour hard-coded into every page means changing the colour touches "
            "every page. The goal of good design is to reduce the amount of code "
            "affected by each design decision."
        ),
    ),
    FixtureRecord(
        key="ko-cognitive-load",
        record_type="knowledge_object",
        title="Cognitive load",
        text=(
            "Symptom: Cognitive load\n"
            "How much a developer must know in order to complete a task.\n"
            "Cognitive load is the second symptom of complexity. An interface that "
            "requires callers to remember to free a returned buffer, or to call "
            "methods in a particular order, raises it. Higher cognitive load means "
            "more time to learn and more chances to introduce bugs."
        ),
    ),
    FixtureRecord(
        key="ko-unknown-unknowns",
        record_type="knowledge_object",
        title="Unknown unknowns",
        text=(
            "Symptom: Unknown unknowns\n"
            "It is not obvious which pieces of code must be modified, or what "
            "information is needed to modify them.\n"
            "Unknown unknowns are the worst of the three symptoms of complexity: "
            "with the other two you at least know what you are facing. Here there "
            "is no way to be sure a change is complete short of reading everything."
        ),
    ),
    FixtureRecord(
        key="ch-modules-deep",
        record_type="chapter",
        title="Modules Should Be Deep",
        text=(
            "The best modules provide powerful functionality behind a simple "
            "interface. Depth is the ratio of the functionality a module offers to "
            "the complexity of the interface it presents. Interface simplicity, not "
            "implementation simplicity, is what matters, because the interface is "
            "the cost every user of the module pays."
        ),
    ),
    FixtureRecord(
        key="ko-deep-module",
        record_type="knowledge_object",
        title="Deep module",
        text=(
            "Principle: Deep module\n"
            "A module whose interface is much simpler than its implementation.\n"
            "A deep module hides a great deal behind a small interface. The Unix "
            "file I/O interface is the canonical example: five system calls stand "
            "in front of scheduling, buffering, permissions and device drivers. "
            "Depth is what makes a module worth having."
        ),
    ),
    FixtureRecord(
        key="ko-shallow-module",
        record_type="knowledge_object",
        title="Shallow module",
        text=(
            "Red flag: Shallow module\n"
            "A module whose interface is as complicated as its implementation.\n"
            "A shallow module costs more to learn than it saves in work. A method "
            "that only sets a field, or a class that merely forwards to another, is "
            "shallow. Classitis — the belief that classes should be small and "
            "numerous — manufactures shallow modules in bulk."
        ),
    ),
    FixtureRecord(
        key="ko-pass-through-method",
        record_type="knowledge_object",
        title="Pass-through method",
        text=(
            "Red flag: Pass-through method\n"
            "A method that does nothing except call another method with much the "
            "same signature.\n"
            "A pass-through method adds an interface without adding functionality, "
            "which is the definition of shallow. It usually means responsibility "
            "has been split between two classes that do not need to be separate."
        ),
    ),
    FixtureRecord(
        key="ko-conjoined-methods",
        record_type="knowledge_object",
        title="Conjoined methods",
        text=(
            "Red flag: Conjoined methods\n"
            "Two methods that cannot be understood independently of each other.\n"
            "If reading one method requires reading another to make sense of it, "
            "the decomposition has failed. Splitting a method is only worthwhile "
            "when the pieces are separately understandable."
        ),
    ),
    FixtureRecord(
        key="ch-information-hiding",
        record_type="chapter",
        title="Information Hiding and Leakage",
        text=(
            "Each module should encapsulate a few pieces of knowledge that "
            "represent design decisions, and those decisions should not appear in "
            "its interface. Information leakage happens when the same knowledge is "
            "reflected in more than one place, so that changing it means changing "
            "all of them."
        ),
    ),
    FixtureRecord(
        key="ko-temporal-decomposition",
        record_type="knowledge_object",
        title="Temporal decomposition",
        text=(
            "Red flag: Temporal decomposition\n"
            "Structure follows the order operations happen in, rather than what "
            "knowledge they need.\n"
            "Temporal decomposition is the most common cause of information "
            "leakage. Splitting a file reader and a file writer into separate "
            "classes because reading happens before writing forces both to know "
            "the file format. Focus on knowledge, not on order."
        ),
    ),
    FixtureRecord(
        key="sec-general-purpose",
        record_type="section",
        title="Somewhat General-Purpose Interfaces",
        text=(
            "Make modules somewhat general-purpose: the interface should be general "
            "enough to support several uses, while the implementation solves the "
            "problem you actually have. A specialised interface tends to push "
            "special-case knowledge outward to every caller."
        ),
    ),
    FixtureRecord(
        key="ch-different-layer",
        record_type="chapter",
        title="Different Layer, Different Abstraction",
        text=(
            "Adjacent layers of a system should provide different abstractions. "
            "When they do not, the extra layer adds cost without adding value. "
            "Pass-through variables, which are threaded through many methods that "
            "do nothing with them, are a symptom of the same problem."
        ),
    ),
    FixtureRecord(
        key="ch-tactical-strategic",
        record_type="chapter",
        title="Working Code Isn't Enough",
        text=(
            "Tactical programming optimises for getting the next feature working; "
            "strategic programming treats a working design as the goal and accepts "
            "that this costs more up front. The investment is roughly ten to twenty "
            "percent of development time, and it pays back within months."
        ),
    ),
    FixtureRecord(
        key="ko-tactical-tornado",
        record_type="knowledge_object",
        title="Tactical tornado",
        text=(
            "Anti-pattern: Tactical tornado\n"
            "A prolific developer who produces features faster than anyone, and "
            "leaves wreckage behind.\n"
            "Management may see a tactical tornado as a hero. Better engineers see "
            "someone whose speed comes from making everyone else's work harder."
        ),
    ),
    FixtureRecord(
        key="ch-design-it-twice",
        record_type="chapter",
        title="Design It Twice",
        text=(
            "Consider at least two radically different options for any major design "
            "decision before choosing. Comparing alternatives teaches you what the "
            "important properties are, and the final choice is usually better than "
            "either candidate. Designers who are used to being right find this "
            "uncomfortable, which is exactly why it is worth doing."
        ),
    ),
    FixtureRecord(
        key="ch-comments-abstractions",
        record_type="chapter",
        title="Comments Describe Things That Aren't Obvious",
        text=(
            "The reason to write a comment is that some information could not be "
            "expressed in the code itself. Comments that restate the code add "
            "length without adding meaning. The best comments describe things at a "
            "different level of detail than the code: lower for what a line does, "
            "higher for what a method is for."
        ),
    ),
    FixtureRecord(
        key="sec-comments-first",
        record_type="section",
        title="Write the Comments First",
        text=(
            "Write the interface comment before writing the body. Doing so turns "
            "documentation into a design tool: a comment that is long or awkward to "
            "write is telling you the abstraction behind it is wrong, at the moment "
            "when it is still cheap to change."
        ),
    ),
    FixtureRecord(
        key="ko-obscurity",
        record_type="knowledge_object",
        title="Obscurity",
        text=(
            "Cause: Obscurity\n"
            "Important information about the system is not obvious to a new "
            "reader.\n"
            "Obscurity is one of the two causes of complexity, alongside "
            "dependencies. It is usually fixed by better naming and better "
            "comments, and sometimes only by a simpler design."
        ),
    ),
    FixtureRecord(
        key="ch-exceptions",
        record_type="chapter",
        title="Define Errors Out of Existence",
        text=(
            "Exception handling is one of the worst sources of complexity, because "
            "every exception a method throws becomes part of its interface. The "
            "best way to reduce the handling is to reduce the number of cases that "
            "count as errors at all — define the situation as normal, and the "
            "special-case code disappears."
        ),
    ),
    FixtureRecord(
        key="ko-exception-masking",
        record_type="knowledge_object",
        title="Exception masking",
        text=(
            "Technique: Exception masking\n"
            "Handle a condition at a low level so higher levels never see it.\n"
            "Exception masking reduces the number of places that must deal with a "
            "condition to one. TCP retransmission is the standard example: a lost "
            "packet is handled entirely inside the transport, and no application "
            "ever hears about it."
        ),
    ),
    FixtureRecord(
        key="ch-consistency",
        record_type="chapter",
        title="Consistency",
        text=(
            "Consistency reduces the cost of learning a system: once you understand "
            "how one part works, the same understanding carries to others. It "
            "applies to names, coding style, interfaces and design patterns, and it "
            "is maintained by writing conventions down and enforcing them."
        ),
    ),
)


# --- the query set ----------------------------------------------------------

QUERIES: tuple[FixtureQuery, ...] = (
    # ---- exact-term: the book's coined vocabulary, verbatim ----
    FixtureQuery(
        query="temporal decomposition",
        kind="exact-term",
        expects=("ko-temporal-decomposition",),
        why="A coined phrase. Nothing about the words suggests their meaning.",
    ),
    FixtureQuery(
        query="pass-through method",
        kind="exact-term",
        expects=("ko-pass-through-method",),
        why="Near-duplicate vocabulary elsewhere ('pass-through variables'), so a "
        "term match has to beat a topical neighbour.",
    ),
    FixtureQuery(
        query="change amplification",
        kind="exact-term",
        expects=("ko-change-amplification", "ch-complexity"),
        why="Named symptom. The complexity chapter summary names it too and is a "
        "fair answer, so both count — an exact-term query with a decoy that is "
        "genuinely relevant is the realistic case.",
    ),
    FixtureQuery(
        query="unknown unknowns",
        kind="exact-term",
        expects=("ko-unknown-unknowns", "ch-complexity"),
        why="Two very common words in an uncommon pairing — the case where an "
        "embedding is most likely to drift to the wrong record.",
    ),
    FixtureQuery(
        query="conjoined methods",
        kind="exact-term",
        expects=("ko-conjoined-methods",),
        why="Rare term whose record shares topic ('methods', 'split') with several "
        "others.",
    ),
    FixtureQuery(
        query="tactical tornado",
        kind="exact-term",
        expects=("ko-tactical-tornado",),
        why="Proper-noun-like coinage; 'tactical' also appears in the chapter "
        "summary, so the exact pair must win.",
    ),
    FixtureQuery(
        query="classitis",
        kind="exact-term",
        expects=("ko-shallow-module",),
        why="A word that appears exactly once in the corpus and almost certainly "
        "never in the embedding model's training data.",
    ),
    # ---- conceptual: paraphrase, no content word shared with the target ----
    FixtureQuery(
        query="a tiny tweak forces me to rewrite dozens of files",
        kind="conceptual",
        expects=("ko-change-amplification",),
        why="Describes the symptom in entirely different words: not one content "
        "word of the query shares a stem with the record.",
    ),
    FixtureQuery(
        query="how many things does a person have to hold in their head before "
        "they can get anything done",
        kind="conceptual",
        expects=("ko-cognitive-load",),
        why="Plain-language restatement with no vocabulary overlap.",
    ),
    FixtureQuery(
        query="a lot of capability reachable through only a handful of entry points",
        kind="conceptual",
        expects=("ko-deep-module", "ch-modules-deep"),
        why="Describes depth without using 'deep', 'module', 'interface' or any "
        "other word either record contains.",
    ),
    FixtureQuery(
        query="is it useful to weigh several rival approaches up front",
        kind="conceptual",
        expects=("ch-design-it-twice",),
        why="Asks the question the chapter answers without using its title words, "
        "or any other word in it.",
    ),
    FixtureQuery(
        query="make the awkward path just another ordinary path",
        kind="conceptual",
        expects=("ch-exceptions", "ko-exception-masking"),
        why="Restates 'define errors out of existence' avoiding 'error', "
        "'exception' and 'handle' entirely.",
    ),
    FixtureQuery(
        query="should i spend effort today to make tomorrow cheaper",
        kind="conceptual",
        expects=("ch-tactical-strategic",),
        why="The investment argument, none of its vocabulary.",
    ),
    # ---- mixed: a real question, part vocabulary and part paraphrase ----
    FixtureQuery(
        query="why is a shallow module bad for the people calling it",
        kind="mixed",
        expects=("ko-shallow-module",),
        why="Names the term, then asks about it in its own words.",
    ),
    FixtureQuery(
        query="comments that just repeat what the code already says",
        kind="mixed",
        expects=("ch-comments-abstractions",),
        why="Shares 'comments' and 'code'; the rest is paraphrase.",
    ),
    FixtureQuery(
        query="information leakage between modules",
        kind="mixed",
        expects=("ch-information-hiding", "ko-temporal-decomposition"),
        why="Exact term plus a general word, and two records legitimately answer.",
    ),
    FixtureQuery(
        query="should an interface be general purpose or specific to my problem",
        kind="mixed",
        expects=("sec-general-purpose",),
        why="Shares 'general-purpose' and 'interface' with the target section.",
    ),
    FixtureQuery(
        query="what makes code obscure to a new reader",
        kind="mixed",
        expects=("ko-obscurity",),
        why="Shares 'obscure/obscurity' as a stem and 'reader'.",
    ),
)


BY_KEY = {record.key: record for record in CORPUS}


# --- measurement ------------------------------------------------------------


@dataclass(frozen=True)
class QueryOutcome:
    """One query under one mode: where each expected record landed."""

    query: FixtureQuery
    mode: SearchMode
    # Expected record key -> its 1-based rank, or None when it did not appear.
    ranks: dict[str, int | None]

    @property
    def best_rank(self) -> int | None:
        found = [rank for rank in self.ranks.values() if rank is not None]
        return min(found) if found else None

    def hit_at(self, k: int) -> bool:
        best = self.best_rank
        return best is not None and best <= k

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first expected record — 0 when none was retrieved.

        MRR is the right summary here because every query has a small number of
        right answers and users read from the top: moving a record from rank 4 to
        rank 1 matters far more than moving it from 40 to 20."""
        best = self.best_rank
        return 1.0 / best if best else 0.0


def summarise(outcomes: Sequence[QueryOutcome], k: int = 5) -> dict[str, float]:
    """Hit@k and MRR over a slice of outcomes, plus the slice's size so a report
    can show what an average was taken over."""
    if not outcomes:
        return {"queries": 0, f"hit@{k}": 0.0, "mrr": 0.0}
    return {
        "queries": len(outcomes),
        f"hit@{k}": sum(outcome.hit_at(k) for outcome in outcomes) / len(outcomes),
        "mrr": sum(outcome.reciprocal_rank for outcome in outcomes) / len(outcomes),
    }


_WORD = re.compile(r"[a-z]+")

# Longest first, so "ing" is tried before "s".
_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    """A crude suffix-stripper, deliberately over-eager.

    Not an attempt at real stemming — it exists so the fixture guards compare
    words the way BM25 will. fastembed's BM25 runs a Snowball stemmer, so a
    "conceptual" query saying `touch` against a record saying `touches` would
    match lexically even though an exact comparison sees two different words.
    Over-stemming here only makes the guard stricter, which is the safe
    direction for a check whose whole job is to stop a query claiming to be
    something it is not.
    """
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Stems of the lower-cased words of 4+ letters — a rough stand-in for
    "content word", used only to check the fixture set's own honesty."""
    return {_stem(word) for word in _WORD.findall(text.lower()) if len(word) >= 4}


# --- harness ----------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """The fixture, materialised: rows in the database and points in Qdrant."""

    book_id: uuid.UUID
    # Fixture key -> the record id its row was given, so a hit can be scored.
    ids: dict[str, uuid.UUID]

    def key_of(self, record_type: RecordType, record_id: uuid.UUID) -> str | None:
        for key, candidate in self.ids.items():
            if candidate == record_id and BY_KEY[key].record_type == record_type:
                return key
        return None


def populate(
    session: Session,
    store: VectorStore,
    book_id: uuid.UUID,
    embedder: EmbeddingProvider,
    sparse_embedder: SparseEmbeddingProvider,
) -> Corpus:
    """Write the fixture's rows, embed their texts, and store both vectors.

    Goes through the real write path (``replace_book_points``) rather than the
    embeddings Stage, because the Stage composes its own text out of row columns
    and the eval needs to embed exactly the text the fixture authored — that text
    *is* the experiment. The rows themselves exist so search can resolve a hit
    back to something; only the embedded text affects retrieval.
    """
    ids: dict[str, uuid.UUID] = {}
    chapter = None
    for record in CORPUS:
        if record.record_type == "chapter":
            row: Chapter | Section | KnowledgeObject = Chapter(
                book_id=book_id, position=len(ids), title=record.title, summary=record.text
            )
            session.add(row)
            session.flush()
            chapter = row
        elif record.record_type == "section":
            assert chapter is not None, "a section fixture must follow a chapter fixture"
            row = Section(
                chapter_id=chapter.id, position=len(ids), title=record.title, summary=record.text
            )
            session.add(row)
            session.flush()
        else:
            row = KnowledgeObject(
                book_id=book_id,
                type="Principle",
                title=record.title,
                content=record.text,
                summary=record.text,
                source_location="fixture",
                confidence=1.0,
                extraction_model="fixture",
                extraction_prompt_version="1",
            )
            session.add(row)
            session.flush()
        ids[record.key] = row.id
    session.commit()

    texts = [record.text for record in CORPUS]
    dense_vectors: list[list[float]] = []
    for start in range(0, len(texts), embedder.max_batch):
        dense_vectors.extend(embedder.embed(texts[start : start + embedder.max_batch]).vectors)
    sparse_vectors = sparse_embedder.embed_documents(texts)

    store.replace_book_points(
        str(book_id),
        [
            VectorRecord(
                id=str(uuid.uuid4()),
                vector=dense,
                sparse=sparse,
                payload={
                    "record_type": record.record_type,
                    "record_id": str(ids[record.key]),
                    "book_id": str(book_id),
                    "text": record.text,
                },
            )
            for record, dense, sparse in zip(CORPUS, dense_vectors, sparse_vectors, strict=True)
        ],
        embedder.model,
        sparse_embedder.recipe,
    )
    return Corpus(book_id=book_id, ids=ids)


def run_query(
    session: Session,
    store: VectorStore,
    corpus: Corpus,
    embedder: EmbeddingProvider,
    sparse_embedder: SparseEmbeddingProvider,
    fixture: FixtureQuery,
    mode: SearchMode,
    limit: int = 10,
) -> QueryOutcome:
    """One fixture query through the real search path, scored."""
    results = search(
        session,
        store,
        embedder,
        fixture.query,
        sparse_embedder=sparse_embedder if mode == "hybrid" else None,
        mode=mode,
        book_id=corpus.book_id,
        limit=limit,
    )
    found: dict[str, int] = {}
    for hit in results.hits:
        key = corpus.key_of(hit.record_type, hit.record_id)
        if key is not None and key not in found:
            found[key] = hit.rank
    return QueryOutcome(
        query=fixture,
        mode=mode,
        ranks={key: found.get(key) for key in fixture.expects},
    )


def compare(
    session: Session,
    store: VectorStore,
    corpus: Corpus,
    embedder: EmbeddingProvider,
    sparse_embedder: SparseEmbeddingProvider,
    limit: int = 10,
) -> list[tuple[QueryOutcome, QueryOutcome]]:
    """Every fixture query, both modes, paired (hybrid, dense)."""
    return [
        (
            run_query(
                session, store, corpus, embedder, sparse_embedder, fixture, "hybrid", limit
            ),
            run_query(
                session, store, corpus, embedder, sparse_embedder, fixture, "dense", limit
            ),
        )
        for fixture in QUERIES
    ]


# --- report -----------------------------------------------------------------


def _rank_cell(outcome: QueryOutcome) -> str:
    """Every expected record and where it landed, `—` for one that never showed."""
    parts = []
    for key, rank in outcome.ranks.items():
        parts.append(f"{key} @{rank}" if rank is not None else f"{key} —")
    return ", ".join(parts)


def _verdict(hybrid: QueryOutcome, dense: QueryOutcome) -> str:
    """Which mode put an expected record higher, comparing best ranks.

    Not-retrieved loses to any rank, and two not-retrieveds are a tie: the metric
    is "how close to the top did the right answer get", and absence is the same
    absence either way."""
    best_hybrid, best_dense = hybrid.best_rank, dense.best_rank
    if best_hybrid == best_dense:
        return "tie"
    if best_dense is None:
        return "**hybrid**"
    if best_hybrid is None:
        return "dense"
    return "**hybrid**" if best_hybrid < best_dense else "dense"


def render_report(
    pairs: Sequence[tuple[QueryOutcome, QueryOutcome]],
    *,
    embedding_model: str,
    sparse_recipe: str,
    limit: int,
) -> str:
    """The eval as Markdown: per-query ranks, per-kind aggregates, and the
    numbers a conclusion would have to be written from."""
    lines = [
        "<!-- Generated by packages/core/tests/test_hybrid_eval.py. Do not edit by hand. -->",
        "",
        f"- Dense embedding model: `{embedding_model}`",
        f"- Sparse recipe: `{sparse_recipe}`",
        f"- Corpus: {len(CORPUS)} fixture records · Queries: {len(QUERIES)} · limit={limit}",
        "",
        "## Per-query ranks",
        "",
        "Rank of each expected record, best first. `—` means it was not in the "
        "top "
        f"{limit}.",
        "",
        "| kind | query | hybrid | dense-only | better |",
        "| --- | --- | --- | --- | --- |",
    ]
    for hybrid, dense in pairs:
        lines.append(
            f"| {hybrid.query.kind} | {hybrid.query.query} | {_rank_cell(hybrid)} "
            f"| {_rank_cell(dense)} | {_verdict(hybrid, dense)} |"
        )

    lines += ["", "## Aggregates", "", "| slice | mode | hit@5 | MRR |", "| --- | --- | --- | --- |"]
    kinds: list[str] = ["exact-term", "conceptual", "mixed"]
    for slice_name, selected in [
        ("all", pairs),
        *[
            (kind, [pair for pair in pairs if pair[0].query.kind == kind])
            for kind in kinds
        ],
    ]:
        for index, mode in enumerate(("hybrid", "dense-only")):
            stats = summarise([pair[index] for pair in selected])
            lines.append(
                f"| {slice_name} (n={stats['queries']:.0f}) | {mode} "
                f"| {stats['hit@5']:.2f} | {stats['mrr']:.3f} |"
            )
    return "\n".join(lines) + "\n"
