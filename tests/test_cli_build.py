import json
from pathlib import Path

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


class FakeEmbeddingClient:
    def __init__(self, base_url: str, model: str, dimension: int) -> None:
        assert base_url == "http://127.0.0.1:8080"
        assert model == "nomic-embed-text-v1.5"
        assert dimension == 3

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        return tuple(EmbeddedUnit(unit, [1.0, 2.0, 3.0]) for unit in units)


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
        "requested_ref": "v2.3.0",
        "resolved_commit": "a" * 40,
        "selected_files": ["README.md"],
        "skipped_files": {},
        "prose_rows": 1,
        "code_rows": 0,
    }
