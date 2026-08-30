import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import typer

from qemer_ingest.artifact import build_artifact
from qemer_ingest.chunking import chunk_units
from qemer_ingest.discovery import discover
from qemer_ingest.embedding import EmbeddingClient
from qemer_ingest.github import GitHubClient
from qemer_ingest.models import BuildReport
from qemer_ingest.parsing import parse_document_with_report
from qemer_ingest.tokenization import TokenizationClient

app = typer.Typer(no_args_is_help=True)
_INCLUDE_OPTION = typer.Option([], "--include")
_OUTPUT_OPTION = typer.Option(..., "--output")
_CHUNK_SIZE_TOKENS_OPTION = typer.Option(2048, "--chunk-size-tokens")
_CHUNK_OVERLAP_TOKENS_OPTION = typer.Option(0, "--chunk-overlap-tokens")
_DOCUMENT_PREFIX_OPTION = typer.Option("", "--document-prefix")


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

        typer.echo(f"Repository: {source.url}")
        typer.echo(f"Requested ref: {source.requested_ref}")
        typer.echo(f"Resolved commit: {source.commit_sha}")
        for path in report.selected:
            typer.echo(f"Selected: {path.as_posix()}")
        for path, reason in report.skipped.items():
            typer.echo(f"Skipped: {path} ({reason})")
        for path in report.explicitly_included:
            typer.echo(f"Explicitly included: {path.as_posix()}")


@app.command()
def build(
    repository_url: str,
    ref: str = typer.Option(..., "--ref"),
    library: str = typer.Option(..., "--library"),
    version: str = typer.Option(..., "--version"),
    embedding_url: str = typer.Option(..., "--embedding-url"),
    embedding_model: str = typer.Option(..., "--embedding-model"),
    embedding_dim: int = typer.Option(..., "--embedding-dim"),
    output: Path = _OUTPUT_OPTION,
    include: list[str] = _INCLUDE_OPTION,
    chunk_size_tokens: int = _CHUNK_SIZE_TOKENS_OPTION,
    chunk_overlap_tokens: int = _CHUNK_OVERLAP_TOKENS_OPTION,
    document_prefix: str = _DOCUMENT_PREFIX_OPTION,
) -> None:
    """Build a local corpus artifact from a resolved repository revision."""
    if not library.strip():
        raise typer.BadParameter("must not be empty", param_hint="--library")
    if not version.strip():
        raise typer.BadParameter("must not be empty", param_hint="--version")
    if embedding_dim <= 0:
        raise typer.BadParameter("must be positive", param_hint="--embedding-dim")
    if chunk_size_tokens <= 0:
        raise typer.BadParameter("must be positive", param_hint="--chunk-size-tokens")
    if chunk_overlap_tokens < 0:
        raise typer.BadParameter(
            "must be non-negative", param_hint="--chunk-overlap-tokens"
        )
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise typer.BadParameter(
            "must be smaller than chunk size in tokens",
            param_hint="--chunk-overlap-tokens",
        )
    if os.path.lexists(output):
        raise typer.BadParameter("must not already exist", param_hint="--output")

    tokenizer = TokenizationClient(embedding_url)
    try:
        asyncio.run(tokenizer.preflight())
        prefix_tokens = (
            asyncio.run(tokenizer.tokenize(document_prefix, add_special=True))
            if document_prefix
            else ()
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--embedding-url") from error
    if len(prefix_tokens) >= chunk_size_tokens:
        raise typer.BadParameter(
            "document prefix leaves no room for source tokens",
            param_hint="--document-prefix",
        )

    github = GitHubClient()
    source = asyncio.run(github.resolve(repository_url, ref))
    with TemporaryDirectory() as temporary_directory:
        repository_root = asyncio.run(
            github.download_archive(source, Path(temporary_directory))
        )
        discovery = discover(repository_root, tuple(include))
        if not discovery.selected:
            raise typer.BadParameter("no documentation files selected")

        parsed_documents = tuple(
            parse_document_with_report(
                repository_root / relative_path,
                source,
                library,
                version,
                repository_root=repository_root,
            )
            for relative_path in discovery.selected
        )
        parsed_units = tuple(
            unit for parsed in parsed_documents for unit in parsed.units
        )
        units = asyncio.run(
            chunk_units(
                parsed_units,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                document_prefix=document_prefix,
            )
        )
        if not units:
            raise typer.BadParameter("selected documentation produced no rows")

        embedding = EmbeddingClient(
            embedding_url,
            embedding_model,
            embedding_dim,
            document_prefix=document_prefix,
        )
        embedded = asyncio.run(embedding.embed_all(units))
        report = BuildReport(
            repository_url=source.url,
            requested_ref=source.requested_ref,
            resolved_commit=source.commit_sha,
            selected_files=discovery.selected,
            skipped_files=discovery.skipped,
            explicitly_included_files=discovery.explicitly_included,
            parser_skips=tuple(
                skip for parsed in parsed_documents for skip in parsed.skipped
            ),
            prose_rows=sum(unit.kind == "prose" for unit in units),
            code_rows=sum(unit.kind == "code" for unit in units),
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            document_prefix=document_prefix,
        )
        build_artifact(
            output,
            library,
            version,
            embedding_model,
            embedding_dim,
            embedded,
            report,
        )

    typer.echo(f"Resolved commit: {source.commit_sha}")
    typer.echo(f"Selected files: {len(discovery.selected)}")
    typer.echo(f"Emitted rows: {len(embedded)}")
    typer.echo(f"Manifest: {output / 'manifest.json'}")


def main() -> None:
    app()
