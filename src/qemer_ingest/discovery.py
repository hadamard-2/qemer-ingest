from pathlib import Path, PurePosixPath

from qemer_ingest.models import DiscoveryReport

_DOCUMENT_EXTENSIONS = {".md", ".rst", ".txt"}
_DOCUMENT_DIRECTORIES = {"docs", "doc", "documentation"}
_EXCLUDED_DIRECTORIES = {".git", "node_modules", "vendor"}
_ROOT_README_NAMES = {"readme", "readme.md", "readme.rst", "readme.txt"}


def discover(root: Path, includes: tuple[str, ...]) -> DiscoveryReport:
    """Select UTF-8 documentation files from an extracted repository."""
    selected: list[Path] = []
    explicitly_included: list[Path] = []
    skipped: dict[str, str] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(root)
        relative_posix = PurePosixPath(relative_path.as_posix())
        path_key = relative_posix.as_posix()
        if _is_excluded(relative_posix):
            skipped[path_key] = "excluded path"
            continue
        is_explicitly_included = _is_included(relative_posix, includes)
        if not _is_selected_by_default(relative_posix) and not is_explicitly_included:
            skipped[path_key] = "outside default documentation paths"
            continue

        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped[path_key] = "not valid UTF-8"
            continue
        selected.append(relative_path)
        if is_explicitly_included:
            explicitly_included.append(relative_path)

    selected.sort(key=lambda path: path.as_posix())
    explicitly_included.sort(key=lambda path: path.as_posix())
    return DiscoveryReport(
        selected=tuple(selected),
        skipped=dict(sorted(skipped.items())),
        explicitly_included=tuple(explicitly_included),
    )


def _is_excluded(path: PurePosixPath) -> bool:
    return any(
        segment.startswith(".") or segment in _EXCLUDED_DIRECTORIES
        for segment in path.parts
    )


def _is_selected_by_default(path: PurePosixPath) -> bool:
    if len(path.parts) == 1 and path.name.casefold() in _ROOT_README_NAMES:
        return True
    return (
        len(path.parts) > 1
        and path.parts[0].casefold() in _DOCUMENT_DIRECTORIES
        and path.suffix.casefold() in _DOCUMENT_EXTENSIONS
    )


def _is_included(path: PurePosixPath, includes: tuple[str, ...]) -> bool:
    return any(path.match(pattern) for pattern in includes)
