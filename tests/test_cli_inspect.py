from pathlib import Path

from typer.testing import CliRunner

from qemer_ingest import cli
from qemer_ingest.models import RepositoryRef


class FakeGitHubClient:
    async def resolve(self, url: str, ref: str) -> RepositoryRef:
        return RepositoryRef(
            url=url,
            owner="example",
            repository="project",
            requested_ref=ref,
            commit_sha="a" * 40,
        )

    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "project-a"
        docs = repository_root / "docs"
        docs.mkdir(parents=True)
        src = repository_root / "src"
        src.mkdir()
        (repository_root / "README.md").write_text("Project overview", encoding="utf-8")
        (docs / "guide.md").write_text("Guide", encoding="utf-8")
        (docs / "broken.md").write_bytes(b"\xff")
        (src / "notes.md").write_text("Requested notes", encoding="utf-8")
        (src / "module.py").write_text("print('not docs')", encoding="utf-8")
        return repository_root


def test_inspect_reports_resolved_documentation_without_creating_output(
    monkeypatch, tmp_path: Path
) -> None:
    def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("inspect reached the embedding client")

    monkeypatch.setattr(cli, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(cli, "EmbeddingClient", unexpected_embedding)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "inspect",
            "https://github.com/example/project",
            "--ref",
            "v1.0.0",
            "--include",
            "src/notes.md",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Repository: https://github.com/example/project" in result.output
    assert "Requested ref: v1.0.0" in result.output
    assert "a" * 40 in result.output
    assert "Selected: README.md" in result.output
    assert "Selected: docs/guide.md" in result.output
    assert "Selected: src/notes.md" in result.output
    assert "Skipped: docs/broken.md (not valid UTF-8)" in result.output
    assert (
        "Skipped: src/module.py (outside default documentation paths)" in result.output
    )
    assert "Explicitly included: src/notes.md" in result.output
    assert "embedding" not in result.output.casefold()
    assert not isinstance(result.exception, AssertionError)
    assert not (tmp_path / "output").exists()
