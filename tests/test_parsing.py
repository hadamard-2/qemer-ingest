from hashlib import sha256
from pathlib import Path

import pytest

from qemer_ingest.models import RepositoryRef
from qemer_ingest.parsing import parse_document

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = RepositoryRef(
    url="https://github.com/example/project",
    owner="example",
    repository="project",
    requested_ref="v1.0.0",
    commit_sha="a" * 40,
)


def expected_id(path: str, ordinal: int) -> str:
    digest = sha256(f"example-lib\0v1.0.0\0{path}\0{ordinal}".encode()).hexdigest()[:16]
    return f"example-lib-v1.0.0-{digest}"


def test_parse_markdown_emits_prose_and_combined_code_per_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(FIXTURES)

    units = parse_document(Path("guide.md"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        ("prose", "Introduction", "Welcome to the guide."),
        ("code", "Introduction", 'print("hello")'),
        ("prose", "Arrays", "Arrays hold ordered values."),
        ("code", "Arrays", "values = [1, 2]\n\nprint(values[0])"),
    ]
    assert [unit.snippet_id for unit in units] == [
        expected_id("guide.md", 1),
        expected_id("guide.md", 1),
        expected_id("guide.md", 2),
        expected_id("guide.md", 2),
    ]
    assert {unit.source_url for unit in units} == {
        f"https://github.com/example/project/blob/{SOURCE.commit_sha}/guide.md"
    }


def test_parse_rst_emits_ordinary_text_and_code_block_per_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(FIXTURES)

    units = parse_document(Path("api.rst"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        ("prose", "API Reference", "The API exposes one callable."),
        ("code", "API Reference", "def answer() -> int:\n    return 42"),
    ]
    assert [unit.snippet_id for unit in units] == [
        expected_id("api.rst", 1),
        expected_id("api.rst", 1),
    ]


def test_parse_rst_separates_a_nested_literal_block_from_list_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(FIXTURES)

    units = parse_document(Path("nested.rst"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        ("prose", "Nested literal", "A list item owns a code sample."),
        ("code", "Nested literal", 'print("nested")'),
    ]


def test_parse_rst_preserves_nested_list_paragraph_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(FIXTURES)

    units = parse_document(Path("nested-prose.rst"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        (
            "prose",
            "Nested prose",
            "First paragraph in the list.\n\nSecond paragraph in the same list item.",
        ),
    ]


def test_parse_rst_keeps_document_introduction_before_named_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(FIXTURES)

    units = parse_document(
        Path("document-introduction.rst"), SOURCE, "example-lib", "v1.0.0"
    )

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        (
            "prose",
            "Project Guide",
            "This introduction applies to the whole document.",
        ),
        ("prose", "Usage", "Use the public API."),
        ("code", "Usage", 'print("usage")'),
    ]


def test_parse_document_uses_repository_relative_identity_across_roots(
    tmp_path: Path,
) -> None:
    repository_path = Path("docs/guide.md")
    roots = (tmp_path / "first-extraction", tmp_path / "second-extraction")
    parsed = []
    for root in roots:
        document = root / repository_path
        document.parent.mkdir(parents=True)
        document.write_text("# Guide\n\nStable prose.\n", encoding="utf-8")
        parsed.append(
            parse_document(
                document,
                SOURCE,
                "example-lib",
                "v1.0.0",
                repository_root=root,
            )
        )

    assert [unit.snippet_id for unit in parsed[0]] == [
        unit.snippet_id for unit in parsed[1]
    ]
    assert {unit.source_url for units in parsed for unit in units} == {
        f"https://github.com/example/project/blob/{SOURCE.commit_sha}/docs/guide.md"
    }


def test_empty_sections_do_not_consume_emitted_section_ordinals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = tmp_path / "guide.rst"
    document.write_text(
        "Guide\n=====\n\nEmpty\n-----\n\nKept\n----\n\nUseful prose.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    units = parse_document(Path("guide.rst"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.title, unit.snippet_id) for unit in units] == [
        ("Kept", expected_id("guide.rst", 1))
    ]


def test_parse_text_uses_file_stem_and_skips_empty_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "overview.TXT").write_text("\nA compact overview.\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text(" \n\t", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    text_units = parse_document(Path("overview.TXT"), SOURCE, "example-lib", "v1.0.0")
    empty_units = parse_document(Path("empty.txt"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in text_units] == [
        ("prose", "overview", "A compact overview."),
    ]
    assert text_units[0].snippet_id == expected_id("overview.TXT", 1)
    assert empty_units == ()


def test_parse_bare_readme_as_plain_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "README").write_text("Repository overview.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    units = parse_document(Path("README"), SOURCE, "example-lib", "v1.0.0")

    assert [(unit.kind, unit.title, unit.text) for unit in units] == [
        ("prose", "README", "Repository overview."),
    ]


def test_parse_document_rejects_unsupported_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "notes.adoc").write_text("Unsupported", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="unsupported document type"):
        parse_document(Path("notes.adoc"), SOURCE, "example-lib", "v1.0.0")
