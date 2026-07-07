"""Booksmart's error taxonomy (API-first).

Every error the pipeline raises on purpose carries a class-level ``retriable``
flag so a Runner can map it to its own retry semantics without parsing message
strings: booksmart-api turns ``retriable=False`` into Inngest's
``NonRetriableError``; the CLI renders one clean line. Vendor SDK exceptions
(anthropic/openai/qdrant network errors and the like) are *not* wrapped — they
pass through untouched, and a Runner treats an unrecognised exception as it
sees fit.

Non-retriable:
- ``ProviderConfigError`` — a Preference conflicts with a Limit, or the
  configuration is otherwise deterministically wrong.
- ``StagePreconditionError`` — the data a stage needs is absent (missing book
  or parsed artifact, unknown scope). Retrying the same call cannot fix it.

Retriable:
- ``ProviderResponseError`` — the model returned nothing usable (a refusal, an
  empty completion, or a response still unparseable after the stage's own
  single retry). A fresh attempt may well succeed.
"""


class BooksmartError(Exception):
    """Base for every error booksmart raises deliberately.

    ``retriable`` is a class-level fact about the failure mode, not an instance
    detail: a whole subclass is either retriable or it is not.
    """

    retriable: bool = False


class ProviderConfigError(BooksmartError, ValueError):
    """A Preference conflicts with a Limit (or the configuration is otherwise
    deterministically wrong) — at provider construction or at the first write
    the model-locked vector collection rejects. Retrying cannot fix it.

    Also a ``ValueError`` so callers that already catch configuration mistakes
    that way keep working.
    """

    retriable = False


class StagePreconditionError(BooksmartError):
    """A stage was asked to run before the data it depends on exists: the book
    row is gone, the parsed markdown artifact is missing, or the scope is
    unknown. A Runner cannot fix this by retrying the same stage."""

    retriable = False


class ProviderResponseError(BooksmartError):
    """A model provider returned no usable result — a refusal, an empty
    completion, or a response still unparseable after the stage's own in-line
    retry. Subsumes the narrower LLMError / ExtractionError / SummaryError,
    which remain as descriptive subclasses. Retriable: a fresh call may
    succeed."""

    retriable = True
