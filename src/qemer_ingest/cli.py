import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import typer

from qemer_ingest.discovery import discover
from qemer_ingest.github import GitHubClient

app = typer.Typer(no_args_is_help=True)
_INCLUDE_OPTION = typer.Option([], "--include")


@app.callback()
def _root() -> None:
    """Build local documentation corpora from public GitHub repositories."""


@app.command()
def inspect(
    repository_url: str,
    ref: str = typer.Option(..., "--ref"),
    include: list[str] = _INCLUDE_OPTION,
) -> None:
    """Show documentation selected from a resolved repository revision."""
    client = GitHubClient()
    source = asyncio.run(client.resolve(repository_url, ref))
    with TemporaryDirectory() as temporary_directory:
        repository_root = asyncio.run(
            client.download_archive(source, Path(temporary_directory))
        )
        report = discover(repository_root, tuple(include))

        typer.echo(f"Resolved commit: {source.commit_sha}")
        for path in report.selected:
            typer.echo(path.as_posix())


def main() -> None:
    app()
