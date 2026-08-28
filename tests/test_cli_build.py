import json
from pathlib import Path
from typing import ClassVar

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


class FakeEmbeddingClient:
    def __init__(self, base_url: str, model: str, dimension: int) -> None:
        assert base_url == "http://127.0.0.1:8080"
        assert model == "nomic-embed-text-v1.5"
        assert dimension == 3

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        return tuple(EmbeddedUnit(unit, [1.0, 2.0, 3.0]) for unit in units)


class RecordingEmbeddingClient(FakeEmbeddingClient):
    calls: ClassVar[list[tuple[DocumentUnit, ...]]] = []

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        self.calls.append(units)
        return await super().embed_all(units)


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
    }


def test_build_report_records_discovery_and_parser_decisions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", ReportingGitHubClient)
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
    }


def test_build_uses_repository_relative_identity_across_extraction_roots(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(cli, "EmbeddingClient", RecordingEmbeddingClient)
    RecordingEmbeddingClient.calls.clear()

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

    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    arguments = build_arguments(tmp_path / "output")
    option_index = arguments.index(invalid_option)
    arguments[option_index + 1] = invalid_value

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code != 0
    assert message in result.output
    assert not isinstance(result.exception, AssertionError)


def test_build_rejects_existing_output_before_external_clients(
    monkeypatch, tmp_path: Path
) -> None:
    def unexpected_client():
        raise AssertionError("existing output reached an external client")

    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    output = tmp_path / "output"
    output.mkdir()

    result = CliRunner().invoke(cli.app, build_arguments(output))

    assert result.exit_code != 0
    assert "must not already exist" in result.output
    assert not isinstance(result.exception, AssertionError)


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
    monkeypatch.setattr(cli, "EmbeddingClient", unexpected_embedding)
    output = tmp_path / "output"

    result = CliRunner().invoke(cli.app, build_arguments(output))

    assert result.exit_code != 0
    assert message in result.output
    assert not isinstance(result.exception, AssertionError)
    assert not output.exists()
