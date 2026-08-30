import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from qemer_ingest.models import BuildReport, EmbeddedUnit


def build_artifact(
    output: Path,
    library: str,
    version: str,
    model: str,
    dimension: int,
    embedded: tuple[EmbeddedUnit, ...],
    report: BuildReport,
) -> Path:
    """Write a complete local corpus artifact, then atomically publish it."""
    if os.path.lexists(output):
        raise FileExistsError(f"output path already exists: {output}")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    _validate_vectors(embedded, dimension)

    archive_name = _archive_name(library, version)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging_name:
        staging = Path(staging_name)
        parquet_path = staging / "corpus.parquet"
        _write_parquet(parquet_path, embedded, dimension)

        archive_path = staging / archive_name
        _write_archive(archive_path, parquet_path)
        parquet_path.unlink()
        archive_bytes = archive_path.read_bytes()
        manifest = {
            "corpora": [
                {
                    "library": library,
                    "version": version,
                    "url": archive_name,
                    "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "bytes": len(archive_bytes),
                    "embedding_model": model,
                    "embedding_dim": dimension,
                    "snippet_count": len(embedded),
                }
            ]
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "build-report.json", _report_payload(report))
        staging.rename(output)

    return output


def _validate_vectors(embedded: tuple[EmbeddedUnit, ...], dimension: int) -> None:
    for item in embedded:
        if len(item.vector) != dimension:
            raise ValueError(
                f"embedding dimension for {item.unit.source_url} is {len(item.vector)}, "
                f"expected {dimension}"
            )


def _archive_name(library: str, version: str) -> str:
    archive_name = f"{library}-{version}.tar.zst"
    archive_path = Path(archive_name)
    if archive_path.is_absolute() or len(archive_path.parts) != 1:
        raise ValueError("archive name must be exactly one relative filename")
    return archive_name


def _write_parquet(
    destination: Path, embedded: tuple[EmbeddedUnit, ...], dimension: int
) -> None:
    vector_type = pa.list_(pa.float32(), list_size=dimension)
    schema = pa.schema(
        [
            pa.field("snippet_id", pa.string(), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("source_url", pa.string()),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", vector_type, nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([item.unit.snippet_id for item in embedded], type=pa.string()),
            pa.array([item.unit.kind for item in embedded], type=pa.string()),
            pa.array([item.unit.title for item in embedded], type=pa.string()),
            pa.array([item.unit.source_url for item in embedded], type=pa.string()),
            pa.array([item.unit.text for item in embedded], type=pa.string()),
            pa.array([item.vector for item in embedded], type=vector_type),
        ],
        schema=schema,
    )
    pq.write_table(table, destination)


def _write_archive(destination: Path, parquet_path: Path) -> None:
    parquet_bytes = parquet_path.read_bytes()
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        member = tarfile.TarInfo("corpus.parquet")
        member.size = len(parquet_bytes)
        tar.addfile(member, io.BytesIO(parquet_bytes))
    destination.write_bytes(zstd.ZstdCompressor().compress(tar_buffer.getvalue()))


def _report_payload(report: BuildReport) -> dict[str, object]:
    return {
        "repository_url": report.repository_url,
        "requested_ref": report.requested_ref,
        "resolved_commit": report.resolved_commit,
        "selected_files": [path.as_posix() for path in report.selected_files],
        "skipped_files": report.skipped_files,
        "explicitly_included_files": [
            path.as_posix() for path in report.explicitly_included_files
        ],
        "parser_skips": [
            {
                "path": skip.path.as_posix(),
                "reason": skip.reason,
                **({"section": skip.section} if skip.section is not None else {}),
            }
            for skip in report.parser_skips
        ],
        "prose_rows": report.prose_rows,
        "code_rows": report.code_rows,
        "chunk_size_tokens": report.chunk_size_tokens,
        "chunk_overlap_tokens": report.chunk_overlap_tokens,
        "document_prefix": report.document_prefix,
    }


def _write_json(destination: Path, payload: dict[str, object]) -> None:
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
