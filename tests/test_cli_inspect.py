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
        (repository_root / "README.md").write_text("Project overview", encoding="utf-8")
        (docs / "guide.md").write_text("Guide", encoding="utf-8")
        return repository_root


def test_inspect_reports_resolved_documentation_without_creating_output(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "GitHubClient", FakeGitHubClient)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        ["inspect", "https://github.com/example/project", "--ref", "v1.0.0"],
    )

    assert result.exit_code == 0, result.output
    assert "a" * 40 in result.output
    assert "README.md" in result.output
    assert "docs/guide.md" in result.output
    assert "embedding" not in result.output.casefold()
    assert not (tmp_path / "output").exists()
