"""The `booksmart config` command and the CLI's settings precedence chain."""

import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booksmart_cli.config import load_settings
from booksmart_cli.main import app


def config_file(home: Path) -> Path:
    return home / "config.toml"


class TestSetAndGet:
    def test_set_get_round_trip(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(app, ["config", "set", "llm_model", "claude-sonnet-5"])
        assert result.exit_code == 0, result.output

        result = runner.invoke(app, ["config", "get", "llm_model"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "claude-sonnet-5"

    def test_set_writes_a_0600_file(self, runner: CliRunner, home: Path) -> None:
        runner.invoke(app, ["config", "set", "llm_model", "claude-sonnet-5"])

        mode = stat.S_IMODE(config_file(home).stat().st_mode)
        assert mode == 0o600

    def test_set_preserves_hand_written_comments(self, runner: CliRunner, home: Path) -> None:
        home.mkdir(parents=True, exist_ok=True)
        config_file(home).write_text("# my note\nllm_model = \"claude-sonnet-5\"\n")

        result = runner.invoke(app, ["config", "set", "llm_provider", "openai"])
        assert result.exit_code == 0, result.output

        assert "# my note" in config_file(home).read_text()

    def test_secret_value_can_come_from_stdin(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(
            app, ["config", "set", "anthropic_api_key"], input="sk-ant-secret\n"
        )
        assert result.exit_code == 0, result.output
        # Confirmation echoes a redacted value, never the key itself.
        assert "sk-ant-secret" not in result.output

        result = runner.invoke(app, ["config", "get", "anthropic_api_key"])
        assert result.stdout.strip() == "sk-ant-secret"

    def test_get_of_unset_field_fails(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(app, ["config", "get", "anthropic_api_key"])

        assert result.exit_code == 1
        assert "not set" in result.stderr

    def test_unknown_field_is_rejected(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(app, ["config", "set", "llm_provder", "openai"])

        assert result.exit_code == 1
        assert "Unknown field" in result.stderr

    def test_invalid_provider_fails_at_set_time(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(app, ["config", "set", "llm_provider", "hal9000"])

        assert result.exit_code == 1
        assert "must be one of" in result.stderr
        assert not config_file(home).exists()


class TestUnset:
    def test_unset_removes_the_value(self, runner: CliRunner, home: Path) -> None:
        runner.invoke(app, ["config", "set", "llm_model", "claude-sonnet-5"])

        result = runner.invoke(app, ["config", "unset", "llm_model"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["config", "get", "llm_model"])
        assert result.exit_code == 1

    def test_unset_of_absent_field_is_a_no_op(self, runner: CliRunner, home: Path) -> None:
        result = runner.invoke(app, ["config", "unset", "llm_model"])

        assert result.exit_code == 0
        assert "nothing to do" in result.stdout


class TestPrecedence:
    def test_booksmart_env_beats_config_file(
        self, runner: CliRunner, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner.invoke(app, ["config", "set", "anthropic_api_key", "sk-from-file"])
        monkeypatch.setenv("BOOKSMART_ANTHROPIC_API_KEY", "sk-from-env")

        result = runner.invoke(app, ["config", "get", "anthropic_api_key"])

        assert result.stdout.strip() == "sk-from-env"

    def test_config_file_beats_conventional_env(
        self, runner: CliRunner, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A deliberate `config set` must win over an ambient shell-profile export.
        runner.invoke(app, ["config", "set", "anthropic_api_key", "sk-from-file"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient")

        result = runner.invoke(app, ["config", "get", "anthropic_api_key"])

        assert result.stdout.strip() == "sk-from-file"

    def test_conventional_env_fills_the_gap(
        self, runner: CliRunner, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient")

        result = runner.invoke(app, ["config", "get", "anthropic_api_key"])

        assert result.stdout.strip() == "sk-ambient"

    def test_load_settings_reflects_the_chain(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home.mkdir(parents=True, exist_ok=True)
        config_file(home).write_text('openai_api_key = "sk-from-file"\n')
        monkeypatch.setenv("GEMINI_API_KEY", "sk-ambient")

        settings = load_settings()

        assert settings.openai_api_key == "sk-from-file"
        assert settings.gemini_api_key == "sk-ambient"
        assert settings.database_url == f"sqlite:///{home / 'booksmart.db'}"
        assert settings.qdrant_path == home / "qdrant"


class TestList:
    def test_list_redacts_secrets_and_shows_sources(
        self, runner: CliRunner, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner.invoke(app, ["config", "set", "anthropic_api_key", "sk-ant-api03-longsecret"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-conventional-ambient")

        result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "sk-ant-api03-longsecret" not in result.stdout
        assert "sk-conventional-ambient" not in result.stdout
        assert "config" in result.stdout  # provenance column
        assert "(unset)" in result.stdout  # unset fields stay discoverable

    def test_loose_file_permissions_warn(self, runner: CliRunner, home: Path) -> None:
        runner.invoke(app, ["config", "set", "llm_model", "claude-sonnet-5"])
        config_file(home).chmod(0o644)

        result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "readable by other users" in result.stderr


class TestMissingKeyRemedy:
    def test_ingest_without_keys_points_at_config_set(
        self,
        runner: CliRunner,
        home: Path,
        monkeypatch: pytest.MonkeyPatch,
        make_pdf,
        add_book,
        tmp_path: Path,
    ) -> None:
        book_id = add_book(make_pdf(tmp_path / "book.pdf"))
        # The fixture selects fake providers; go back to the real default.
        monkeypatch.delenv("BOOKSMART_LLM_PROVIDER")
        monkeypatch.delenv("BOOKSMART_EMBEDDING_PROVIDER")

        result = runner.invoke(app, ["ingest", book_id])

        assert result.exit_code == 1
        assert "Anthropic API key is not set" in result.stderr
        assert "booksmart config set anthropic_api_key" in result.stderr
