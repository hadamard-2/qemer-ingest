from pathlib import Path

from qemer_ingest.discovery import discover


def test_discover_selects_conventional_docs_and_explicit_includes(tmp_path: Path) -> None:
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
