from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from docutils import nodes
from docutils.core import publish_doctree
from markdown_it import MarkdownIt

from qemer_ingest.models import DocumentUnit, RepositoryRef


def parse_document(
    path: Path, source: RepositoryRef, library: str, version: str
) -> tuple[DocumentUnit, ...]:
    """Parse one repository-relative documentation file into Qemer units."""
    contents = path.read_text(encoding="utf-8")
    suffix = path.suffix.casefold()

    if suffix == ".md":
        sections = _parse_markdown(contents, path.stem)
    elif suffix == ".rst":
        sections = _parse_rst(contents, path.stem)
    elif suffix == ".txt":
        text = contents.strip()
        sections = ((path.stem, text, ""),) if text else ()
    else:
        raise ValueError(f"unsupported document type: {path.suffix}")

    relative_path = path.as_posix()
    source_url = (
        f"https://github.com/{source.owner}/{source.repository}/blob/"
        f"{source.commit_sha}/{relative_path}"
    )
    return _make_units(sections, source_url, relative_path, library, version)


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
        if prose_text or code_text:
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

    finalize()
    return tuple(sections)


def _parse_rst(contents: str, fallback_title: str) -> tuple[Section, ...]:
    document = publish_doctree(contents)
    sections = list(document.findall(nodes.section))
    if sections:
        return tuple(_rst_section(section, fallback_title) for section in sections)
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
    return "".join(prose), code


def _make_units(
    sections: tuple[Section, ...],
    source_url: str,
    relative_path: str,
    library: str,
    version: str,
) -> tuple[DocumentUnit, ...]:
    units: list[DocumentUnit] = []
    for ordinal, (title, prose, code) in enumerate(sections, start=1):
        snippet_id = _snippet_id(library, version, relative_path, ordinal)
        if prose:
            units.append(DocumentUnit(snippet_id, "prose", title, source_url, prose))
        if code:
            units.append(DocumentUnit(snippet_id, "code", title, source_url, code))
    return tuple(units)


def _snippet_id(library: str, version: str, relative_path: str, ordinal: int) -> str:
    digest = sha256(
        f"{library}\0{version}\0{relative_path}\0{ordinal}".encode()
    ).hexdigest()[:16]
    return f"{library}-{version}-{digest}"


type Section = tuple[str, str, str]
