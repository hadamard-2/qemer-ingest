from pathlib import Path

from qemer_ingest.discovery import discover


def test_discover_selects_conventional_docs_and_explicit_includes(
    tmp_path: Path,
) -> None:
    files = {
        "README.md": "Project overview",
        "docs/guide.md": "Guide",
        "doc/api.rst": "API\n===",
        "documentation/intro.txt": "Introduction",
        "src/notes.md": "Implementation notes",
        ".github/guide.md": "Hidden metadata",
        "node_modules/readme.md": "Dependency documentation",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    default_report = discover(tmp_path, ())
    included_report = discover(tmp_path, ("src/notes.md",))

    assert [path.as_posix() for path in default_report.selected] == [
        "README.md",
        "doc/api.rst",
        "docs/guide.md",
        "documentation/intro.txt",
    ]
    assert [path.as_posix() for path in included_report.selected] == [
        "README.md",
        "doc/api.rst",
        "docs/guide.md",
        "documentation/intro.txt",
        "src/notes.md",
    ]


def test_discover_skips_non_utf8_files_in_posix_order(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a-first.md").write_bytes(b"\xff")
    nested_docs = docs / "z-last"
    nested_docs.mkdir()
    (nested_docs / "deep.md").write_bytes(b"\xff")
    documentation = tmp_path / "documentation"
    documentation.mkdir()
    (documentation / "a-intro.md").write_bytes(b"\xff")

    report = discover(tmp_path, ())

    assert report.selected == ()
    assert list(report.skipped) == [
        "docs/a-first.md",
        "docs/z-last/deep.md",
        "documentation/a-intro.md",
    ]
    assert report.skipped == {
        "docs/a-first.md": "not valid UTF-8",
        "docs/z-last/deep.md": "not valid UTF-8",
        "documentation/a-intro.md": "not valid UTF-8",
    }


def test_discover_includes_do_not_bypass_excluded_paths(tmp_path: Path) -> None:
    excluded_files = (
        ".private/notes.md",
        ".git/guide.md",
        "vendor/manual.md",
    )
    for relative_path in excluded_files:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Excluded documentation", encoding="utf-8")

    report = discover(tmp_path, excluded_files)

    assert report.selected == ()
    assert report.skipped == {}
