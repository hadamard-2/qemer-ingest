from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from docutils import nodes
from docutils.core import publish_doctree
from markdown_it import MarkdownIt

from qemer_ingest.models import (
    DocumentUnit,
    ParsedDocument,
    ParserSkip,
    RepositoryRef,
)


def parse_document(
    path: Path,
    source: RepositoryRef,
    library: str,
    version: str,
    *,
    repository_root: Path | None = None,
) -> tuple[DocumentUnit, ...]:
    """Parse one documentation file using its repository-relative identity."""
    return parse_document_with_report(
        path,
        source,
        library,
        version,
        repository_root=repository_root,
    ).units


def parse_document_with_report(
    path: Path,
    source: RepositoryRef,
    library: str,
    version: str,
    *,
    repository_root: Path | None = None,
) -> ParsedDocument:
    """Parse one document and retain empty-file and empty-section decisions."""
    relative_path = _repository_relative_path(path, repository_root)
    contents = path.read_text(encoding="utf-8")
    suffix = path.suffix.casefold()

    if suffix == ".md":
        sections = _parse_markdown(contents, path.stem) if contents.strip() else ()
    elif suffix == ".rst":
        sections = _parse_rst(contents, path.stem) if contents.strip() else ()
    elif suffix == ".txt" or (not suffix and path.name.casefold() == "readme"):
        text = contents.strip()
        sections = ((path.stem, text, ""),) if text else ()
    else:
        raise ValueError(f"unsupported document type: {path.suffix}")

    if not sections:
        return ParsedDocument((), (ParserSkip(Path(relative_path), "empty file"),))
    source_url = (
        f"https://github.com/{source.owner}/{source.repository}/blob/"
        f"{source.commit_sha}/{relative_path}"
    )
    units, skipped = _make_units(sections, source_url, relative_path, library, version)
    return ParsedDocument(units, skipped)


def _parse_markdown(contents: str, fallback_title: str) -> tuple[Section, ...]:
    parser = MarkdownIt("commonmark")
    sections: list[Section] = []
    title = fallback_title
    prose: list[str] = []
    code: list[str] = []
    has_heading = False
    tokens = parser.parse(contents)
    index = 0

    def finalize() -> None:
        prose_text = "\n\n".join(part.strip() for part in prose if part.strip()).strip()
        code_text = "\n\n".join(part.strip() for part in code if part.strip()).strip()
        sections.append((title, prose_text, code_text))

    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open":
            if has_heading or prose or code:
                finalize()
            has_heading = True
            title_token = tokens[index + 1]
            title = title_token.content.strip() or fallback_title
            prose = []
            code = []
            index += 2
        elif token.type == "inline":
            prose.append(token.content)
        elif token.type == "fence":
            code.append(token.content)
        index += 1

    if has_heading or prose or code:
        finalize()
    return tuple(sections)


def _parse_rst(contents: str, fallback_title: str) -> tuple[Section, ...]:
    document = publish_doctree(contents)
    sections = list(document.findall(nodes.section))
    if sections:
        title_node = next(
            (child for child in document.children if isinstance(child, nodes.title)),
            None,
        )
        title = title_node.astext().strip() if title_node else fallback_title
        introduction = _rst_children(
            [
                child
                for child in document.children
                if child is not title_node and not isinstance(child, nodes.section)
            ],
            title,
        )
        parsed_sections = [
            _rst_section(section, fallback_title) for section in sections
        ]
        if introduction[1] or introduction[2]:
            parsed_sections.insert(0, introduction)
        return tuple(parsed_sections)
    title_node = next(
        (child for child in document.children if isinstance(child, nodes.title)), None
    )
    title = title_node.astext().strip() if title_node else fallback_title
    children = [child for child in document.children if child is not title_node]
    return (_rst_children(children, title),)


def _rst_section(section: nodes.section, fallback_title: str) -> Section:
    title_node = next(
        (child for child in section.children if isinstance(child, nodes.title)), None
    )
    title = title_node.astext().strip() if title_node else fallback_title
    children = [child for child in section.children if child is not title_node]
    return _rst_children(children, title)


def _rst_children(children: Iterable[nodes.Node], title: str) -> Section:
    prose: list[str] = []
    code: list[str] = []
    for child in children:
        text, child_code = _rst_content(child)
        if text.strip():
            prose.append(text.strip())
        code.extend(child_code)
    return (title, "\n\n".join(prose).strip(), "\n\n".join(code).strip())


def _rst_content(node: nodes.Node) -> tuple[str, list[str]]:
    if isinstance(node, nodes.section):
        return "", []
    if isinstance(node, (nodes.literal_block, nodes.doctest_block)):
        return "", [node.astext()]
    if isinstance(node, nodes.Text):
        return str(node), []

    prose: list[str] = []
    code: list[str] = []
    for child in node.children:
        text, child_code = _rst_content(child)
        prose.append(text)
        code.extend(child_code)
    separator = "" if isinstance(node, (nodes.inline, nodes.paragraph)) else "\n\n"
    return separator.join(text for text in prose if text), code


def _make_units(
    sections: tuple[Section, ...],
    source_url: str,
    relative_path: str,
    library: str,
    version: str,
) -> tuple[tuple[DocumentUnit, ...], tuple[ParserSkip, ...]]:
    units: list[DocumentUnit] = []
    skipped: list[ParserSkip] = []
    ordinal = 0
    for title, prose, code in sections:
        if not prose and not code:
            skipped.append(
                ParserSkip(Path(relative_path), "empty section", section=title)
            )
            continue
        ordinal += 1
        snippet_id = _snippet_id(library, version, relative_path, ordinal)
        if prose:
            units.append(DocumentUnit(snippet_id, "prose", title, source_url, prose))
        if code:
            units.append(DocumentUnit(snippet_id, "code", title, source_url, code))
    return tuple(units), tuple(skipped)


def _snippet_id(library: str, version: str, relative_path: str, ordinal: int) -> str:
    digest = sha256(
        f"{library}\0{version}\0{relative_path}\0{ordinal}".encode()
    ).hexdigest()[:16]
    return f"{library}-{version}-{digest}"


def _repository_relative_path(path: Path, repository_root: Path | None) -> str:
    if repository_root is None:
        if path.is_absolute():
            raise ValueError("absolute document paths require a repository root")
        relative_path = path
    else:
        relative_path = path.relative_to(repository_root)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("document path must be repository-relative")
    return relative_path.as_posix()


type Section = tuple[str, str, str]
