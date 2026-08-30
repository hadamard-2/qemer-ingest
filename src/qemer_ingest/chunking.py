from qemer_ingest.models import DocumentUnit
from qemer_ingest.tokenization import TokenizationClient


async def chunk_units(
    units: tuple[DocumentUnit, ...],
    *,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    document_prefix: str,
) -> tuple[DocumentUnit, ...]:
    _validate_options(chunk_size_tokens, chunk_overlap_tokens)

    output: list[DocumentUnit] = []
    for unit in units:
        try:
            chunks = await _chunk_unit(
                unit,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                document_prefix=document_prefix,
            )
        except ValueError as error:
            raise ValueError(f"{error} for {unit.source_url}") from error
        output.extend(chunks)
    return tuple(output)


def _validate_options(chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
    if chunk_size_tokens <= 0:
        raise ValueError("token size must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            "token overlap must be non-negative and smaller than token size"
        )


async def _chunk_unit(
    unit: DocumentUnit,
    *,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    document_prefix: str,
) -> tuple[DocumentUnit, ...]:
    payload_tokens = await tokenizer.tokenize(
        document_prefix + unit.text, add_special=True
    )
    if len(payload_tokens) <= chunk_size_tokens:
        return (unit,)

    prefix_tokens = await tokenizer.tokenize(document_prefix, add_special=True)
    if len(prefix_tokens) >= chunk_size_tokens:
        raise ValueError("document prefix leaves no room for source tokens")

    source_tokens = await tokenizer.tokenize(unit.text, add_special=False)
    children: list[DocumentUnit] = []
    start = 0
    while start < len(source_tokens):
        end, text = await _longest_fitting_slice(
            source_tokens,
            start=start,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            document_prefix=document_prefix,
        )
        if end == start:
            raise ValueError("token budget leaves no room for source tokens")

        child = DocumentUnit(
            f"{unit.snippet_id}-{unit.kind}-{len(children) + 1:03d}",
            unit.kind,
            unit.title,
            unit.source_url,
            text,
        )
        final_tokens = await tokenizer.tokenize(
            document_prefix + child.text, add_special=True
        )
        if len(final_tokens) > chunk_size_tokens:
            raise ValueError("token slice exceeds token size")
        children.append(child)

        if end == len(source_tokens):
            break
        next_start = end - chunk_overlap_tokens
        if next_start <= start:
            raise ValueError("token overlap leaves no room for new source tokens")
        start = next_start

    return tuple(children)


async def _longest_fitting_slice(
    source_tokens: tuple[int, ...],
    *,
    start: int,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    document_prefix: str,
) -> tuple[int, str]:
    for end in range(len(source_tokens), start, -1):
        text = await tokenizer.detokenize(source_tokens[start:end])
        payload_tokens = await tokenizer.tokenize(
            document_prefix + text, add_special=True
        )
        if len(payload_tokens) <= chunk_size_tokens:
            return end, text
    return start, ""
