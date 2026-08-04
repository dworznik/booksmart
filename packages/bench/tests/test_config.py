"""Where the harness reads truth from, and where it keeps its corpora."""

from collections.abc import Callable
from pathlib import Path

import pytest

from booksmart_bench.config import (
    config_snapshot,
    corpus_home,
    load_settings,
    resolve_assets,
)
from booksmart_bench.errors import BenchError
from booksmart_core.config import Settings


class TestResolveAssets:
    def test_explicit_path_wins_over_the_environment(
        self, make_assets: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chosen = make_assets("chosen")
        monkeypatch.setenv("BOOKSMART_BENCH_ASSETS", str(make_assets("ambient")))

        assert resolve_assets(chosen) == chosen

    def test_environment_is_used_when_no_flag_is_given(
        self, make_assets: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ambient = make_assets("ambient")
        monkeypatch.setenv("BOOKSMART_BENCH_ASSETS", str(ambient))

        assert resolve_assets(None) == ambient

    def test_unset_is_an_error_naming_both_ways_to_set_it(self) -> None:
        with pytest.raises(BenchError) as excinfo:
            resolve_assets(None)

        message = str(excinfo.value)
        assert "--assets" in message
        assert "BOOKSMART_BENCH_ASSETS" in message

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(BenchError, match="does not exist"):
            resolve_assets(tmp_path / "nowhere")

    def test_a_directory_without_truth_is_rejected(
        self, make_assets: Callable[..., Path]
    ) -> None:
        """Pointing at the wrong directory should fail here, not five stages
        later with an empty result set."""
        not_assets = make_assets("not-assets", truth=False)

        with pytest.raises(BenchError, match="truth/"):
            resolve_assets(not_assets)

    def test_the_resolved_path_is_absolute(
        self, make_assets: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run files record where truth came from; a relative path would depend
        on the working directory a verb happened to be invoked from."""
        assets = make_assets("assets")
        monkeypatch.chdir(assets.parent)

        assert resolve_assets(Path("assets")).is_absolute()


class TestConfigSnapshot:
    def test_is_stable_for_the_same_settings(self) -> None:
        settings = Settings(embedding_model="text-embedding-3-small")

        assert config_snapshot(settings) == config_snapshot(settings)

    def test_changes_when_the_embedding_model_changes(self) -> None:
        """A corpus embedded with one model cannot answer for another."""
        one = config_snapshot(Settings(embedding_model="text-embedding-3-small"))
        other = config_snapshot(Settings(embedding_model="text-embedding-3-large"))

        assert one != other

    def test_changes_when_the_llm_model_changes(self) -> None:
        one = config_snapshot(Settings(llm_model="a-model"))
        other = config_snapshot(Settings(llm_model="another-model"))

        assert one != other

    def test_ignores_settings_that_cannot_change_a_corpus(self) -> None:
        """Keys and store locations move between machines and runs; they say
        nothing about what a corpus contains, so they must not fragment it."""
        plain = Settings()
        relocated = Settings(
            database_url="sqlite:///somewhere/else.db",
            storage_root=Path("/var/tmp/elsewhere"),
            qdrant_path=Path("/var/tmp/qdrant"),
            openai_api_key="sk-rotated",
        )

        assert config_snapshot(plain) == config_snapshot(relocated)

    def test_names_the_providers_so_a_directory_can_be_read_at_a_glance(self) -> None:
        snapshot = config_snapshot(Settings(llm_provider="anthropic", embedding_provider="openai"))

        assert snapshot.startswith("anthropic-openai-")

    def test_is_a_safe_single_path_segment(self) -> None:
        """Model names carry dots, slashes and colons upstream; the snapshot is
        a directory name."""
        snapshot = config_snapshot(Settings(llm_model="publisher/model:v1.5"))

        assert "/" not in snapshot
        assert ":" not in snapshot


class TestLoadSettings:
    def test_reads_prefixed_environment_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BOOKSMART_EMBEDDING_PROVIDER", "fake")

        assert load_settings().embedding_provider == "fake"

    def test_unset_fields_keep_cores_defaults(self) -> None:
        assert load_settings().llm_provider == Settings().llm_provider

    def test_an_unknown_variable_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only Settings fields are read; a stray BOOKSMART_* name in the shell
        is not this harness's business."""
        monkeypatch.setenv("BOOKSMART_HOME", "/somewhere")

        assert load_settings() == Settings()

    def test_the_config_file_is_not_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A benchmark's configuration must be visible in the command that ran
        it, never inherited from the user's CLI home."""
        monkeypatch.setenv("BOOKSMART_HOME", str(Path.home() / ".booksmart"))

        assert load_settings().embedding_model is None


class TestCorpusHome:
    def test_lives_under_the_home_dir_in_a_snapshot_named_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv("BOOKSMART_BENCH_HOME", str(home))
        settings = Settings()

        resolved = corpus_home(settings)

        assert resolved == home / config_snapshot(settings)

    def test_defaults_to_a_dot_directory_beside_the_users_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BOOKSMART_BENCH_HOME", raising=False)

        assert corpus_home(Settings()).parent == Path.home() / ".booksmart-bench"

    def test_two_configurations_never_share_a_corpus(self, tmp_path: Path) -> None:
        one = corpus_home(Settings(embedding_model="text-embedding-3-small"))
        other = corpus_home(Settings(embedding_model="text-embedding-3-large"))

        assert one != other

    def test_resolving_does_not_create_anything(self, tmp_path: Path) -> None:
        """`score` and `report` are pure; asking where a corpus would live must
        not leave a directory behind."""
        resolved = corpus_home(Settings())

        assert not resolved.exists()
