"""The docs site's CLI reference is generated from the Typer app, so it cannot
drift from the CLI it documents (hooks/cli_reference.py, website/cli.md).
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import typer

from booksmart_cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_hook() -> ModuleType:
    path = REPO_ROOT / "hooks" / "cli_reference.py"
    spec = importlib.util.spec_from_file_location("cli_reference_hook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def _walk(command: Any, path: str) -> list[tuple[str, Any]]:
    # Duck-typed: Typer vendors its own click, so `isinstance(..., click.Group)`
    # is False for every Typer group and would make this walk silently vacuous.
    found = [(path, command)]
    for name, sub in getattr(command, "commands", {}).items():
        found.extend(_walk(sub, f"{path} {name}"))
    return found


def test_every_command_and_option_appears() -> None:
    rendered = hook.render(app)
    commands = _walk(typer.main.get_command(app), "booksmart")
    assert len(commands) > 10  # the walk really descended into the groups
    for path, command in commands:
        # The root renders under the page's own heading; every subcommand gets one.
        if path != "booksmart":
            assert f"`{path}`" in rendered, path
        for param in command.params:
            if param.param_type_name == "option":
                for opt in param.opts:
                    assert opt in rendered, f"{path} {opt}"


def test_a_new_command_appears_with_no_doc_edit() -> None:
    demo = typer.Typer()
    demo.add_typer(app, name="booksmart")

    @demo.command()
    def frobnicate(
        depth: int = typer.Option(3, help="How hard to frobnicate."),
    ) -> None:
        """Frobnicate every book."""

    rendered = hook.render(demo, name="demo")
    assert "`demo frobnicate`" in rendered
    assert "Frobnicate every book." in rendered
    assert "--depth" in rendered


class _Page:
    def __init__(self, src_uri: str) -> None:
        self.file = type("File", (), {"src_uri": src_uri})()


def test_hook_replaces_the_marker_on_the_cli_page() -> None:
    markdown = f"# CLI\n\nIntro prose.\n\n{hook.MARKER}\n"
    out = hook.on_page_markdown(markdown, page=_Page("cli.md"))
    assert hook.MARKER not in out
    assert out.startswith("# CLI\n\nIntro prose.")
    assert "`booksmart add`" in out


def test_hook_leaves_other_pages_alone() -> None:
    markdown = f"# Home\n\n{hook.MARKER}\n"
    assert hook.on_page_markdown(markdown, page=_Page("index.md")) == markdown


def test_cli_page_without_the_marker_fails_the_build() -> None:
    from mkdocs.exceptions import PluginError

    with pytest.raises(PluginError):
        hook.on_page_markdown("# CLI\n", page=_Page("cli.md"))


def test_generated_headings_sit_below_the_page_title() -> None:
    rendered = hook.render(app)
    headings: list[Any] = [line for line in rendered.splitlines() if line.startswith("#")]
    assert headings
    assert all(line.startswith("##") for line in headings)
