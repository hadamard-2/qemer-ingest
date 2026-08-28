from qemer_ingest.chunking import chunk_units
from qemer_ingest.models import DocumentUnit


def unit(kind: str = "prose", text: str = "short text") -> DocumentUnit:
    return DocumentUnit(
        "numpy-2.3-abc",
        kind,
        "Arrays",
        "https://example.test/README.md",
        text,
    )


def test_chunk_units_keeps_a_unit_at_or_below_the_limit() -> None:
    original = unit(text="abcdefgh")
    assert chunk_units((original,), chunk_size=8, chunk_overlap=0) == (original,)


def test_chunk_units_splits_oversized_units_with_stable_child_metadata() -> None:
    chunks = chunk_units(
        (unit(text="abcdefghijklmnopqrst"),), chunk_size=8, chunk_overlap=2
    )
    assert [
        (chunk.snippet_id, chunk.kind, chunk.title, chunk.source_url, chunk.text)
        for chunk in chunks
    ] == [
        (
            "numpy-2.3-abc-prose-001",
            "prose",
            "Arrays",
            "https://example.test/README.md",
            "abcdefgh",
        ),
        (
            "numpy-2.3-abc-prose-002",
            "prose",
            "Arrays",
            "https://example.test/README.md",
            "ghijklmn",
        ),
        (
            "numpy-2.3-abc-prose-003",
            "prose",
            "Arrays",
            "https://example.test/README.md",
            "mnopqrst",
        ),
    ]


def test_chunk_units_uses_kind_qualified_ids_for_shared_parent_ids() -> None:
    chunks = chunk_units(
        (unit("prose", "abcdefghij"), unit("code", "klmnopqrst")),
        chunk_size=8,
        chunk_overlap=0,
    )
    assert [chunk.snippet_id for chunk in chunks] == [
        "numpy-2.3-abc-prose-001",
        "numpy-2.3-abc-prose-002",
        "numpy-2.3-abc-code-001",
        "numpy-2.3-abc-code-002",
    ]
