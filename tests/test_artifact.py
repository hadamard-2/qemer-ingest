import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from qemer_ingest.models import BuildReport, DocumentUnit, EmbeddedUnit


def make_embedded() -> tuple[EmbeddedUnit, ...]:
    return (
        EmbeddedUnit(
            DocumentUnit(
                "numpy-2.3-001",
                "prose",
                "Overview",
                "https://github.com/numpy/numpy/blob/commit/README.md",
                "NumPy is for arrays.",
            ),
            [1.0, 2.0, 3.0],
        ),
        EmbeddedUnit(
            DocumentUnit(
                "numpy-2.3-002",
                "code",
                "Example",
                "https://github.com/numpy/numpy/blob/commit/docs/example.md",
                "import numpy as np",
            ),
            [4.0, 5.0, 6.0],
        ),
    )


def make_report() -> BuildReport:
    return BuildReport(
        requested_ref="v2.3.0",
        resolved_commit="a" * 40,
        selected_files=(Path("README.md"), Path("docs/example.md")),
        skipped_files={"src/notes.md": "outside default documentation paths"},
        prose_rows=1,
        code_rows=1,
    )


def test_build_artifact_writes_the_local_qemer_contract(tmp_path: Path) -> None:
    from qemer_ingest.artifact import build_artifact

    output = tmp_path / "numpy-2.3.0"
    result = build_artifact(
        output,
        "numpy",
        "2.3.0",
        "nomic-embed-text-v1.5",
        3,
        make_embedded(),
        make_report(),
    )

    archive_name = "numpy-2.3.0.tar.zst"
    archive = output / archive_name
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert result == output
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        archive_name,
        "build-report.json",
    }
    assert manifest == {
        "library": "numpy",
        "version": "2.3.0",
        "url": archive_name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "bytes": archive.stat().st_size,
        "embedding_model": "nomic-embed-text-v1.5",
        "embedding_dim": 3,
        "snippet_count": 2,
    }

    tar_bytes = zstd.ZstdDecompressor().decompress(archive.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        assert tar.getnames() == ["corpus.parquet"]
        parquet_member = tar.extractfile("corpus.parquet")
        assert parquet_member is not None
        table = pq.read_table(pa.BufferReader(parquet_member.read()))

    assert table.schema == pa.schema(
        [
            pa.field("snippet_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("title", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=3)),
        ]
    )
    assert table.to_pylist() == [
        {
            "snippet_id": "numpy-2.3-001",
            "kind": "prose",
            "title": "Overview",
            "source_url": "https://github.com/numpy/numpy/blob/commit/README.md",
            "text": "NumPy is for arrays.",
            "vector": [1.0, 2.0, 3.0],
        },
        {
            "snippet_id": "numpy-2.3-002",
            "kind": "code",
            "title": "Example",
            "source_url": "https://github.com/numpy/numpy/blob/commit/docs/example.md",
            "text": "import numpy as np",
            "vector": [4.0, 5.0, 6.0],
        },
    ]


def test_build_artifact_rejects_an_existing_output(tmp_path: Path) -> None:
    from qemer_ingest.artifact import build_artifact

    output = tmp_path / "already-exists"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_artifact(
            output,
            "numpy",
            "2.3.0",
            "nomic-embed-text-v1.5",
            3,
            make_embedded(),
            make_report(),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_artifact_rejects_wrong_vector_width_with_source_url(
    tmp_path: Path,
) -> None:
    from qemer_ingest.artifact import build_artifact

    invalid = EmbeddedUnit(make_embedded()[0].unit, [1.0, 2.0])

    with pytest.raises(ValueError, match=invalid.unit.source_url):
        build_artifact(
            tmp_path / "numpy-2.3.0",
            "numpy",
            "2.3.0",
            "nomic-embed-text-v1.5",
            3,
            (invalid,),
            make_report(),
        )
