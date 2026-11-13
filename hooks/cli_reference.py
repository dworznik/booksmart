"""MkDocs hook: generate the CLI reference from the Typer app at build time.

``website/cli.md`` holds the hand-written framing and a marker; this hook swaps
the marker for Markdown that Typer renders from ``booksmart_cli.main:app``. The
reference is never authored, so a command or option added to the CLI reaches
the published page on the next build with no documentation edit.
"""

from typing import Any

import typer
from mkdocs.exceptions import PluginError
from typer.cli import get_docs_for_click
from typer.core import MARKUP_MODE_KEY

PAGE = "cli.md"
MARKER = "<!-- booksmart-cli-reference -->"


def render(app: typer.Typer, name: str = "booksmart") -> str:
    """Markdown for every command and option of ``app``, headings from ``##`` down
    so they nest under the page's own title."""
    command = typer.main.get_command(app)
    ctx = typer.Context(command)
    # Typer's own `typer ... utils docs` passes the markup mode the same way, so
    # rich-formatted help renders as it would in the terminal.
    ctx.obj = {MARKUP_MODE_KEY: getattr(app, "rich_markup_mode", None)}
    docs = get_docs_for_click(obj=command, ctx=ctx, indent=1, name=name, title="Commands")
    return docs.strip() + "\n"


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    if page.file.src_uri != PAGE:
        return markdown
    if MARKER not in markdown:
        # Fail loudly: without the marker the page would ship with no reference.
        raise PluginError(f"{PAGE} must contain {MARKER} where the CLI reference goes")
    from booksmart_cli.main import app

    return markdown.replace(MARKER, render(app))
