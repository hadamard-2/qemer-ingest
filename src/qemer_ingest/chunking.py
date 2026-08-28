from langchain_text_splitters import RecursiveCharacterTextSplitter

from qemer_ingest.models import DocumentUnit

_SEPARATORS = ("\n\n", "\n", " ", "")


def chunk_units(
    units: tuple[DocumentUnit, ...], *, chunk_size: int, chunk_overlap: int
) -> tuple[DocumentUnit, ...]:
    _validate_options(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=list(_SEPARATORS),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    output: list[DocumentUnit] = []
    for unit in units:
        output.extend(_chunk_unit(unit, splitter, chunk_size, chunk_overlap))
    return tuple(output)


def _validate_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk overlap must be non-negative and smaller than chunk size"
        )


def _chunk_unit(
    unit: DocumentUnit,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[DocumentUnit, ...]:
    if len(unit.text) <= chunk_size:
        return (unit,)

    split_texts = tuple(text for text in splitter.split_text(unit.text) if text.strip())
    texts = _enforce_overlap(split_texts, chunk_size, chunk_overlap)
    return tuple(
        DocumentUnit(
            f"{unit.snippet_id}-{unit.kind}-{ordinal:03d}",
            unit.kind,
            unit.title,
            unit.source_url,
            text,
        )
        for ordinal, text in enumerate(texts, start=1)
    )


def _enforce_overlap(
    texts: tuple[str, ...], chunk_size: int, chunk_overlap: int
) -> tuple[str, ...]:
    if chunk_overlap == 0 or not texts:
        return texts

    pending = list(texts)
    first = pending.pop(0)
    while len(first) < chunk_overlap and pending:
        text = pending.pop(0)
        available = chunk_size - len(first)
        first += text[:available]
        if len(text) > available:
            pending.insert(0, text[available:])

    output = [first]
    for text in pending:
        required_prefix = output[-1][-chunk_overlap:]
        if text.startswith(required_prefix):
            output.append(text)
            continue

        remaining = text
        while remaining:
            required_prefix = output[-1][-chunk_overlap:]
            payload_size = chunk_size - len(required_prefix)
            payload = remaining[:payload_size]
            output.append(required_prefix + payload)
            remaining = remaining[payload_size:]

    return tuple(output)
