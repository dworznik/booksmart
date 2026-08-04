"""The two locations the harness needs, and the key that separates corpora.

**Assets** — the ground-truth checkout. It lives outside this repo (truth is
authored against real books and versioned privately), so there is no sensible
default and none is invented: `--assets` or `BOOKSMART_BENCH_ASSETS`, or an
error that names both.

**Corpus home** — ``~/.booksmart-bench/<config-snapshot>/``. Ingest is the only
expensive verb and recall is cheap, so the charter is ingest-once-recall-often:
a corpus is reused until something that would invalidate it moves. The snapshot
is exactly that "something" — the models and the prompt versions that decide
what ends up stored. Everything else about a configuration (where the database
file sits, which key is current) can change freely without splitting a corpus.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from booksmart_core.config import Settings
from booksmart_core.extraction import EXTRACTION_PROMPT_VERSION
from booksmart_core.parsing import EXTRACTION_VERSION
from booksmart_core.profile import PROFILE_PROMPT_VERSION
from booksmart_core.summaries import SUMMARY_PROMPT_VERSION

from booksmart_bench.errors import AssetsNotFoundError

ASSETS_ENV = "BOOKSMART_BENCH_ASSETS"
HOME_ENV = "BOOKSMART_BENCH_HOME"

# The Settings fields that decide what a corpus *contains*. Anything absent from
# this list is deliberately absent: locations and credentials describe where a
# corpus lives and how it was reached, not what is in it.
#
# Add to this list whenever core gains a setting that changes what ingest
# writes — a sparse-retrieval recipe being the obvious next one. A field that
# belongs here and is missing does not fail loudly: two incompatible corpora
# quietly share a directory and the second silently scores against the first.
CORPUS_DEFINING_FIELDS = (
    "llm_provider",
    "llm_model",
    "llm_reasoning_effort",
    "embedding_provider",
    "embedding_model",
)

_UNSAFE_IN_A_PATH = re.compile(r"[^a-z0-9]+")


def resolve_assets(explicit: Path | None) -> Path:
    """The assets checkout, as an absolute path, verified to be one."""
    ambient = os.environ.get(ASSETS_ENV)
    candidate = explicit if explicit is not None else (Path(ambient) if ambient else None)
    if candidate is None:
        raise AssetsNotFoundError(
            "No benchmark assets given. Pass --assets <path> or set "
            f"{ASSETS_ENV} to your assets checkout."
        )

    resolved = candidate.expanduser().resolve()
    if not resolved.is_dir():
        raise AssetsNotFoundError(f"Assets path does not exist (or is not a directory): {resolved}")
    if not (resolved / "truth").is_dir():
        raise AssetsNotFoundError(
            f"{resolved} has no truth/ directory, so it is not an assets checkout."
        )
    return resolved


def snapshot_inputs(settings: Settings) -> dict[str, str | None]:
    """Everything the snapshot is computed from — the models a run uses and the
    pipeline versions stamped on what it writes. Exposed because run files
    record it: a score is only comparable against another run of the same shape.
    """
    inputs: dict[str, str | None] = {
        field: _as_text(getattr(settings, field)) for field in CORPUS_DEFINING_FIELDS
    }
    inputs["extraction_version"] = EXTRACTION_VERSION
    inputs["extraction_prompt_version"] = EXTRACTION_PROMPT_VERSION
    inputs["summary_prompt_version"] = SUMMARY_PROMPT_VERSION
    inputs["profile_prompt_version"] = PROFILE_PROMPT_VERSION
    return inputs


def config_snapshot(settings: Settings) -> str:
    """A stable directory name for one pipeline configuration.

    Readable head, precise tail: the two provider names so a `ls` of the home
    dir means something, then a digest of the full input set so two
    configurations that differ anywhere at all never collide.
    """
    inputs = snapshot_inputs(settings)
    digest = hashlib.blake2b(
        json.dumps(inputs, sort_keys=True).encode(), digest_size=4
    ).hexdigest()
    return f"{_slug(settings.llm_provider)}-{_slug(settings.embedding_provider)}-{digest}"


def load_settings() -> Settings:
    """The pipeline configuration for this invocation, from the environment.

    Deliberately thinner than the CLI's chain: no config file, no vendor
    conventional variables, only ``BOOKSMART_<FIELD>``. The harness is driven
    from a shell or CI where a benchmark's configuration should be visible in
    the command that ran it — a value inherited from a user's home config would
    silently change what a run means, and runs are meant to be comparable.
    """
    provided = {
        name: os.environ[f"BOOKSMART_{name.upper()}"]
        for name in Settings.model_fields
        if f"BOOKSMART_{name.upper()}" in os.environ
    }
    return Settings.model_validate(provided)


def default_home() -> Path:
    """Where corpora live: ``$BOOKSMART_BENCH_HOME`` or ``~/.booksmart-bench``."""
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".booksmart-bench")


def corpus_home(settings: Settings) -> Path:
    """This configuration's corpus directory. Pure — creates nothing."""
    return default_home() / config_snapshot(settings)


def _as_text(value: object) -> str | None:
    return None if value is None else str(value)


def _slug(value: str) -> str:
    return _UNSAFE_IN_A_PATH.sub("-", value.lower()).strip("-") or "unset"
