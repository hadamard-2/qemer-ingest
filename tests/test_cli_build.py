import json
import re
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self

import pytest
from typer.testing import CliRunner

from qemer_ingest import cli
from qemer_ingest.models import DocumentUnit, EmbeddedUnit, RepositoryRef


class FakeGitHubClient:
    async def resolve(self, url: str, ref: str) -> RepositoryRef:
        return RepositoryRef(
            url=url,
            owner="numpy",
            repository="numpy",
            requested_ref=ref,
            commit_sha="a" * 40,
        )

    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        repository_root.mkdir()
        (repository_root / "README.md").write_text(
            "# NumPy\n\nNumPy provides multidimensional arrays.\n", encoding="utf-8"
        )
        return repository_root


class LongReadmeGitHubClient(FakeGitHubClient):
    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        repository_root.mkdir()
        (repository_root / "README.md").write_text(
            "# Overflow\n\nabcdefghijklmnopqrst\n", encoding="utf-8"
        )
        return repository_root


class ReportingGitHubClient(FakeGitHubClient):
    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        docs = repository_root / "docs"
        docs.mkdir(parents=True)
        src = repository_root / "src"
        src.mkdir()
        (repository_root / "README.md").write_text(
            "# Empty\n\n# Overview\n\nNumPy provides arrays.\n",
            encoding="utf-8",
        )
        (docs / "empty.txt").write_text(" \n", encoding="utf-8")
        (docs / "broken.md").write_bytes(b"\xff")
        (src / "notes.md").write_text(
            "Requested implementation notes.", encoding="utf-8"
        )
        (src / "module.py").write_text("print('not docs')", encoding="utf-8")
        return repository_root


class NoFilesGitHubClient(FakeGitHubClient):
    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        src = repository_root / "src"
        src.mkdir(parents=True)
        (src / "module.py").write_text("print('not docs')", encoding="utf-8")
        return repository_root


class NoUnitsGitHubClient(FakeGitHubClient):
    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        repository_root.mkdir()
        (repository_root / "README.md").write_text(" \n", encoding="utf-8")
        return repository_root


class FakeTokenizationClient:
    entered = 0
    exited = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        assert self.base_url == "http://127.0.0.1:8080"

    async def __aenter__(self) -> Self:
        FakeTokenizationClient.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        FakeTokenizationClient.exited += 1

    async def preflight(self) -> None:
        return None

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        special = (-1,) if add_special else ()
        return special + tuple(ord(character) for character in content)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in tokens)


class ConfiguredTokenizationClient(FakeTokenizationClient):
    prefix = "search_document: "
    prefix_token = (0x110000,)

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        special = (-1,) if add_special else ()
        if content.startswith(self.prefix):
            content = content.removeprefix(self.prefix)
            return (
                special
                + self.prefix_token
                + tuple(ord(character) for character in content)
            )
        return special + tuple(ord(character) for character in content)


class EmptyPrefixSpecialTokenizationClient(FakeTokenizationClient):
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if content == "" and add_special:
            return (-1,)
        return await super().tokenize(content, add_special=add_special)


class FakeEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        document_prefix: str = "",
    ) -> None:
        assert base_url == "http://127.0.0.1:8080"
        assert model == "nomic-embed-text-v1.5"
        assert dimension == 3
        assert document_prefix == ""

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        return tuple(EmbeddedUnit(unit, [1.0, 2.0, 3.0]) for unit in units)


class RecordingEmbeddingClient(FakeEmbeddingClient):
    calls: ClassVar[list[tuple[DocumentUnit, ...]]] = []
    prefixes: ClassVar[list[str]] = []

    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        document_prefix: str = "",
    ) -> None:
        self.prefixes.append(document_prefix)
        if document_prefix:
            assert base_url == "http://127.0.0.1:8080"
            assert model == "nomic-embed-text-v1.5"
            assert dimension == 3
        else:
            super().__init__(base_url, model, dimension, document_prefix)

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        self.calls.append(units)
        return await super().embed_all(units)


@pytest.fixture(autouse=True)
def reset_fake_tokenization_lifecycle() -> None:
    FakeTokenizationClient.entered = 0
    FakeTokenizationClient.exited = 0


def build_arguments(output: Path) -> list[str]:
    return [
        "build",
        "https://github.com/numpy/numpy",
        "--ref",
        "v2.3.0",
        "--library",
        "numpy",
        "--version",
        "2.3.0",
        "--embedding-url",
        "http://127.0.0.1:8080",
        "--embedding-model",
        "nomic-embed-text-v1.5",
        "--embedding-dim",
        "3",
        "--output",
        str(output),
    ]


def test_build_writes_local_artifact_from_resolved_repository(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(cli, "TokenizationClient", FakeTokenizationClient)
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbeddingClient, raising=False)
    output = tmp_path / "numpy-2.3.0"

    result = CliRunner().invoke(
        cli.app,
        [
            "build",
            "https://github.com/numpy/numpy",
            "--ref",
            "v2.3.0",
            "--library",
            "numpy",
            "--version",
            "2.3.0",
            "--embedding-url",
            "http://127.0.0.1:8080",
            "--embedding-model",
            "nomic-embed-text-v1.5",
            "--embedding-dim",
            "3",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeTokenizationClient.entered == 1
    assert FakeTokenizationClient.exited == 1
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "numpy-2.3.0.tar.zst",
        "build-report.json",
    }
    report = json.loads((output / "build-report.json").read_text(encoding="utf-8"))
    assert report == {
        "repository_url": "https://github.com/numpy/numpy",
        "requested_ref": "v2.3.0",
        "resolved_commit": "a" * 40,
        "selected_files": ["README.md"],
        "skipped_files": {},
        "explicitly_included_files": [],
        "parser_skips": [],
        "prose_rows": 1,
        "code_rows": 0,
        "chunk_size_tokens": 2048,
        "chunk_overlap_tokens": 0,
        "document_prefix": "",
    }


def test_build_report_records_discovery_and_parser_decisions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", ReportingGitHubClient)
    monkeypatch.setattr(cli, "TokenizationClient", FakeTokenizationClient)
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbeddingClient)
    output = tmp_path / "numpy-report"

    result = CliRunner().invoke(
        cli.app,
        [
            "build",
            "https://github.com/numpy/numpy",
            "--ref",
            "v2.3.0",
            "--library",
            "numpy",
            "--version",
            "2.3.0",
            "--embedding-url",
            "http://127.0.0.1:8080",
            "--embedding-model",
            "nomic-embed-text-v1.5",
            "--embedding-dim",
            "3",
            "--output",
            str(output),
            "--include",
            "src/notes.md",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((output / "build-report.json").read_text(encoding="utf-8"))
    assert report == {
        "repository_url": "https://github.com/numpy/numpy",
        "requested_ref": "v2.3.0",
        "resolved_commit": "a" * 40,
        "selected_files": ["README.md", "docs/empty.txt", "src/notes.md"],
        "skipped_files": {
            "docs/broken.md": "not valid UTF-8",
            "src/module.py": "outside default documentation paths",
        },
        "explicitly_included_files": ["src/notes.md"],
        "parser_skips": [
            {
                "path": "README.md",
                "reason": "empty section",
                "section": "Empty",
            },
            {"path": "docs/empty.txt", "reason": "empty file"},
        ],
        "prose_rows": 2,
        "code_rows": 0,
        "chunk_size_tokens": 2048,
        "chunk_overlap_tokens": 0,
        "document_prefix": "",
    }


def test_build_uses_repository_relative_identity_across_extraction_roots(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(cli, "TokenizationClient", FakeTokenizationClient)
    monkeypatch.setattr(cli, "EmbeddingClient", RecordingEmbeddingClient)
    RecordingEmbeddingClient.calls.clear()
    RecordingEmbeddingClient.prefixes.clear()

    for ordinal in (1, 2):
        output = tmp_path / f"output-{ordinal}"
        result = CliRunner().invoke(
            cli.app,
            [
                "build",
                "https://github.com/numpy/numpy",
                "--ref",
                "v2.3.0",
                "--library",
                "numpy",
                "--version",
                "2.3.0",
                "--embedding-url",
                "http://127.0.0.1:8080",
                "--embedding-model",
                "nomic-embed-text-v1.5",
                "--embedding-dim",
                "3",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output

    first = RecordingEmbeddingClient.calls[0][0]
    second = RecordingEmbeddingClient.calls[1][0]
    assert first.snippet_id == second.snippet_id
    assert first.source_url == (
        f"https://github.com/numpy/numpy/blob/{'a' * 40}/README.md"
    )
    assert second.source_url == first.source_url


def test_build_chunks_before_embedding_and_records_the_token_policy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", LongReadmeGitHubClient)
    monkeypatch.setattr(cli, "TokenizationClient", ConfiguredTokenizationClient)
    monkeypatch.setattr(cli, "EmbeddingClient", RecordingEmbeddingClient)
    RecordingEmbeddingClient.calls.clear()
    RecordingEmbeddingClient.prefixes.clear()
    output = tmp_path / "numpy-chunked"

    result = CliRunner().invoke(
        cli.app,
        build_arguments(output)
        + [
            "--chunk-size-tokens",
            "8",
            "--chunk-overlap-tokens",
            "2",
            "--document-prefix",
            "search_document: ",
        ],
    )

    assert result.exit_code == 0, result.output
    embedded = RecordingEmbeddingClient.calls[0]
    digest_input = "numpy\x002.3.0\x00README.md\x001"
    digest = sha256(digest_input.encode()).hexdigest()[:16]
    parent_id = f"numpy-2.3.0-{digest}"
    assert [unit.snippet_id for unit in embedded] == [
        f"{parent_id}-prose-001",
        f"{parent_id}-prose-002",
        f"{parent_id}-prose-003",
        f"{parent_id}-prose-004",
        f"{parent_id}-prose-005",
    ]
    report = json.loads((output / "build-report.json").read_text())
    assert report["chunk_size_tokens"] == 8
    assert report["chunk_overlap_tokens"] == 2
    assert report["document_prefix"] == "search_document: "
    assert RecordingEmbeddingClient.prefixes == ["search_document: "]
    assert all("search_document: " not in unit.text for unit in embedded)


def test_build_help_describes_token_chunking_options() -> None:
    result = CliRunner().invoke(cli.app, ["build", "--help"])
    assert result.exit_code == 0
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert "--chunk-size-tokens" in output
    assert "--chunk-overlap-tokens" in output
    assert "--document-prefix" in output
    assert "2048" in output
    assert "0" in output
    assert "--chunk-size " not in output
    assert "--chunk-overlap " not in output


@pytest.mark.parametrize(
    ("invalid_option", "invalid_value", "message"),
    (
        ("--library", " ", "must not be empty"),
        ("--version", " ", "must not be empty"),
        ("--embedding-dim", "0", "must be positive"),
    ),
)
def test_build_rejects_invalid_preflight_before_external_clients(
    monkeypatch,
    tmp_path: Path,
    invalid_option: str,
    invalid_value: str,
    message: str,
) -> None:
    def unexpected_client():
        raise AssertionError("invalid preflight reached an external client")

    monkeypatch.setattr(cli, "TokenizationClient", unexpected_client)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    arguments = build_arguments(tmp_path / "output")
    option_index = arguments.index(invalid_option)
    arguments[option_index + 1] = invalid_value

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code != 0
    assert message in result.output
    assert not isinstance(result.exception, AssertionError)
    assert FakeTokenizationClient.entered == 0


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--chunk-size-tokens", "0", "must be positive"),
        ("--chunk-overlap-tokens", "-1", "must be non-negative"),
    ),
)
def test_build_rejects_invalid_token_options_before_external_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    def unexpected_client() -> None:
        raise AssertionError("invalid chunk option constructed a GitHub client")

    monkeypatch.setattr(cli, "TokenizationClient", unexpected_client)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    arguments = build_arguments(tmp_path / "output") + [option, value]
    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code != 0
    assert message in result.output
    assert FakeTokenizationClient.entered == 0


def test_build_rejects_overlap_equal_to_chunk_size_before_external_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_client() -> None:
        raise AssertionError("invalid chunk option constructed a GitHub client")

    monkeypatch.setattr(cli, "TokenizationClient", unexpected_client)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    result = CliRunner().invoke(
        cli.app,
        build_arguments(tmp_path / "output")
        + ["--chunk-size-tokens", "8", "--chunk-overlap-tokens", "8"],
    )

    assert result.exit_code != 0
    assert "must be smaller than chunk size" in result.output
    assert FakeTokenizationClient.entered == 0


def test_build_rejects_existing_output_before_external_clients(
    monkeypatch, tmp_path: Path
) -> None:
    def unexpected_client():
        raise AssertionError("existing output reached an external client")

    monkeypatch.setattr(cli, "TokenizationClient", unexpected_client)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    output = tmp_path / "output"
    output.mkdir()

    result = CliRunner().invoke(cli.app, build_arguments(output))

    assert result.exit_code != 0
    assert "must not already exist" in result.output
    assert not isinstance(result.exception, AssertionError)
    assert FakeTokenizationClient.entered == 0


@pytest.mark.parametrize(
    ("github_client", "message"),
    (
        (NoFilesGitHubClient, "no documentation files selected"),
        (NoUnitsGitHubClient, "selected documentation produced no rows"),
    ),
)
def test_build_fails_before_embedding_when_selection_cannot_emit_rows(
    monkeypatch, tmp_path: Path, github_client, message: str
) -> None:
    def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("an empty build reached the embedding client")

    monkeypatch.setattr(cli, "GitHubClient", github_client)
    monkeypatch.setattr(cli, "TokenizationClient", FakeTokenizationClient)
    monkeypatch.setattr(cli, "EmbeddingClient", unexpected_embedding)
    output = tmp_path / "output"

    result = CliRunner().invoke(cli.app, build_arguments(output))

    assert result.exit_code != 0
    assert message in result.output
    assert not isinstance(result.exception, AssertionError)
    assert not output.exists()


def test_build_token_preflight_fails_before_github_and_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingTokenizationClient(FakeTokenizationClient):
        async def preflight(self) -> None:
            raise ValueError("token endpoint /detokenize failed")

    def unexpected_github_client() -> None:
        raise AssertionError("failed token preflight constructed a GitHub client")

    monkeypatch.setattr(cli, "TokenizationClient", FailingTokenizationClient)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_github_client)
    output = tmp_path / "output"

    result = CliRunner().invoke(cli.app, build_arguments(output))

    assert result.exit_code != 0
    assert "token endpoint /detokenize failed" in result.output
    assert not isinstance(result.exception, AssertionError)
    assert FakeTokenizationClient.entered == 1
    assert FakeTokenizationClient.exited == 1
    assert not output.exists()


def test_build_rejects_a_prefix_without_source_room_before_github(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_github_client() -> None:
        raise AssertionError("prefix-only token budget constructed a GitHub client")

    def unexpected_embedding(*_args, **_kwargs) -> None:
        raise AssertionError("prefix-only token budget reached the embedding client")

    monkeypatch.setattr(cli, "TokenizationClient", FakeTokenizationClient)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_github_client)
    monkeypatch.setattr(cli, "EmbeddingClient", unexpected_embedding)
    output = tmp_path / "output"

    result = CliRunner().invoke(
        cli.app,
        build_arguments(output)
        + ["--chunk-size-tokens", "8", "--document-prefix", "abcdefg"],
    )

    assert result.exit_code != 0
    assert "document prefix" in result.output
    assert "source tokens" in result.output
    assert not isinstance(result.exception, AssertionError)
    assert FakeTokenizationClient.entered == 1
    assert FakeTokenizationClient.exited == 1
    assert not output.exists()


def test_build_counts_empty_prefix_special_tokens_before_github(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_github_client() -> None:
        raise AssertionError("empty-prefix token budget constructed a GitHub client")

    monkeypatch.setattr(cli, "TokenizationClient", EmptyPrefixSpecialTokenizationClient)
    monkeypatch.setattr(cli, "GitHubClient", unexpected_github_client)
    output = tmp_path / "output"

    result = CliRunner().invoke(
        cli.app,
        build_arguments(output) + ["--chunk-size-tokens", "1"],
    )

    assert result.exit_code != 0
    assert "document prefix" in result.output
    assert "source tokens" in result.output
    assert not isinstance(result.exception, AssertionError)
    assert FakeTokenizationClient.entered == 1
    assert FakeTokenizationClient.exited == 1
    assert not output.exists()
