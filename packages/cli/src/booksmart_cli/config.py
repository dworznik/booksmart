"""The CLI's configuration: one precedence chain, and the ``config`` commands.

Core's ``Settings`` is a plain model that reads nothing ambient, so the CLI owns
the entire story of where a value comes from. Every field resolves through the
same chain, highest first:

1. ``BOOKSMART_<FIELD>`` environment variable — explicit targeting, wins always.
2. ``config.toml`` in the home dir — what ``booksmart config set`` writes.
3. The vendor's conventional environment variable (``ANTHROPIC_API_KEY`` /
   ``OPENAI_API_KEY`` / ``GEMINI_API_KEY``) — API keys only. Below the file on
   purpose: these are ambient (exported for a dozen tools), and a deliberate
   ``config set`` must beat whatever happens to be in the shell profile.
4. Defaults — the home-dir locations (SQLite file, ``storage/``, embedded
   Qdrant) or the field's built-in default.

``config list`` shows the winning source per field, so "why is this value
winning?" is always answerable. The file is created ``0600`` (it holds keys)
and written with tomlkit so hand-edits and comments survive round-trips.
"""

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import tomlkit
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from booksmart_core.config import Settings
from booksmart_core.llm import DEFAULT_EMBEDDING_MODELS, DEFAULT_MODELS

from booksmart_cli.errors import CliError, handle_errors

SECRET_FIELDS = frozenset({"anthropic_api_key", "openai_api_key", "gemini_api_key"})

# The SDKs' conventional variables, resolved here (never by core or the SDKs)
# so the whole chain lives in this module.
CONVENTIONAL_ENV = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
}

# Fields whose set-time validation goes beyond their type: catching a typo'd
# provider at `config set` beats catching it at the first ingest.
_PROVIDER_CHOICES = {
    "llm_provider": DEFAULT_MODELS,
    "embedding_provider": DEFAULT_EMBEDDING_MODELS,
}


def default_home() -> Path:
    """The CLI's home directory: ``$BOOKSMART_HOME`` or ``~/.booksmart``."""
    return Path(os.environ.get("BOOKSMART_HOME") or Path.home() / ".booksmart")


def config_path(home: Path) -> Path:
    return home / "config.toml"


@dataclass(frozen=True)
class ResolvedValue:
    value: object
    source: str  # "env" | "config" | "conventional-env" | "default"


def _read_config_doc(path: Path) -> tomlkit.TOMLDocument:
    if not path.exists():
        return tomlkit.document()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        typer.secho(
            f"warning: {path} is readable by other users (mode {mode:03o}); "
            f"consider `chmod 600 {path}`",
            fg=typer.colors.YELLOW,
            err=True,
        )
    return tomlkit.parse(path.read_text())


def _write_config_doc(path: Path, doc: tomlkit.TOMLDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.write_text(tomlkit.dumps(doc))


def resolve(home: Path) -> dict[str, ResolvedValue]:
    """Every Settings field with its effective value and winning source."""
    file_values = {key: value.unwrap() for key, value in _read_config_doc(config_path(home)).items()}
    unknown = set(file_values) - set(Settings.model_fields)
    if unknown:
        raise CliError(
            f"Unknown field(s) in {config_path(home)}: {', '.join(sorted(unknown))}"
        )

    resolved: dict[str, ResolvedValue] = {}
    for name in Settings.model_fields:
        env_var = f"BOOKSMART_{name.upper()}"
        conventional = CONVENTIONAL_ENV.get(name)
        if env_var in os.environ:
            resolved[name] = ResolvedValue(os.environ[env_var], "env")
        elif name in file_values:
            resolved[name] = ResolvedValue(file_values[name], "config")
        elif conventional is not None and conventional in os.environ:
            resolved[name] = ResolvedValue(os.environ[conventional], "conventional-env")
        else:
            resolved[name] = ResolvedValue(Settings.model_fields[name].default, "default")

    # Home-dir location defaults (the CLI's no-server shape). Embedded Qdrant
    # only when the user didn't explicitly point at a server or another path.
    if resolved["database_url"].source == "default":
        resolved["database_url"] = ResolvedValue(f"sqlite:///{home / 'booksmart.db'}", "default")
    if resolved["storage_root"].source == "default":
        resolved["storage_root"] = ResolvedValue(home / "storage", "default")
    if resolved["qdrant_url"].source == "default" and resolved["qdrant_path"].source == "default":
        resolved["qdrant_path"] = ResolvedValue(home / "qdrant", "default")

    return resolved


def load_settings(home: Path | None = None) -> Settings:
    """Resolve the full chain into the explicit ``Settings`` core requires."""
    home = home or default_home()
    home.mkdir(parents=True, exist_ok=True)
    try:
        return Settings.model_validate(
            {name: item.value for name, item in resolve(home).items()}
        )
    except ValidationError as exc:
        raise CliError(f"Invalid configuration: {exc}") from exc


# --- the `booksmart config` command group ------------------------------------

config_app = typer.Typer(
    help="Read and persist configuration (~/.booksmart/config.toml).",
    no_args_is_help=True,
)
_console = Console()


def _require_field(field: str) -> None:
    if field not in Settings.model_fields:
        raise CliError(
            f"Unknown field {field!r}; valid fields: {', '.join(Settings.model_fields)}"
        )


def _validate_value(field: str, value: str) -> None:
    choices = _PROVIDER_CHOICES.get(field)
    if choices is not None and value not in choices:
        raise CliError(f"{field} must be one of {', '.join(sorted(choices))}; got {value!r}")
    try:
        Settings(**{field: value})  # type: ignore[arg-type]
    except ValidationError as exc:
        first = exc.errors()[0]
        raise CliError(f"Invalid value for {field}: {first['msg']}") from exc


def _read_value_interactively(field: str) -> str:
    if sys.stdin.isatty():
        value = typer.prompt(f"Value for {field}", hide_input=field in SECRET_FIELDS)
    else:
        value = sys.stdin.read().strip()
    if not value:
        raise CliError("No value provided.")
    return str(value)


def _redact(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}…{value[-4:]}"


@config_app.command("set")
@handle_errors
def config_set(
    field: Annotated[str, typer.Argument(help="A Settings field, e.g. anthropic_api_key.")],
    value: Annotated[
        Optional[str],
        typer.Argument(
            help="The value. Omit to enter it via hidden prompt (or piped stdin), "
            "keeping secrets out of shell history."
        ),
    ] = None,
) -> None:
    """Persist a value to config.toml (validated immediately)."""
    _require_field(field)
    if value is None:
        value = _read_value_interactively(field)
    _validate_value(field, value)
    home = default_home()
    home.mkdir(parents=True, exist_ok=True)
    doc = _read_config_doc(config_path(home))
    doc[field] = value
    _write_config_doc(config_path(home), doc)
    shown = _redact(value) if field in SECRET_FIELDS else value
    _console.print(f"Set [bold]{field}[/bold] = {shown}")


@config_app.command("get")
@handle_errors
def config_get(
    field: Annotated[str, typer.Argument(help="A Settings field.")],
) -> None:
    """Print the effective value (after the whole precedence chain), unredacted."""
    _require_field(field)
    item = resolve(default_home())[field]
    if item.value is None:
        typer.secho(f"{field} is not set", err=True)
        raise typer.Exit(1)
    typer.echo(str(item.value))


@config_app.command("unset")
@handle_errors
def config_unset(
    field: Annotated[str, typer.Argument(help="A Settings field.")],
) -> None:
    """Remove a value from config.toml (lower layers of the chain take over)."""
    _require_field(field)
    path = config_path(default_home())
    doc = _read_config_doc(path)
    if field not in doc:
        _console.print(f"[bold]{field}[/bold] is not set in {path}; nothing to do.")
        return
    del doc[field]
    _write_config_doc(path, doc)
    _console.print(f"Unset [bold]{field}[/bold]")


@config_app.command("list")
@handle_errors
def config_list() -> None:
    """Every field with its effective value and where it came from."""
    table = Table("field", "value", "source")
    for name, item in resolve(default_home()).items():
        if item.value is None:
            table.add_row(name, "[dim](unset)[/dim]", "")
            continue
        value = str(item.value)
        if name in SECRET_FIELDS:
            value = _redact(value)
        table.add_row(name, value, item.source)
    _console.print(table)
