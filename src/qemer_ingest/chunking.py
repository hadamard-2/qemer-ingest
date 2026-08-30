from qemer_ingest.models import DocumentUnit
from qemer_ingest.tokenization import TokenizationClient

_MAX_REFINEMENT_ATTEMPTS = 8


async def chunk_units(
    units: tuple[DocumentUnit, ...],
    *,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    document_prefix: str,
) -> tuple[DocumentUnit, ...]:
    _validate_options(chunk_size_tokens, chunk_overlap_tokens)

    try:
        prefix_tokens = await tokenizer.tokenize(document_prefix, add_special=True)
    except ValueError as error:
        if units:
            raise ValueError(f"{error} for {units[0].source_url}") from error
        raise
    if len(prefix_tokens) >= chunk_size_tokens:
        raise ValueError("document prefix leaves no room for source tokens")

    output: list[DocumentUnit] = []
    for unit in units:
        try:
            chunks = await _chunk_unit(
                unit,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                document_prefix=document_prefix,
                prefix_token_count=len(prefix_tokens),
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
    prefix_token_count: int,
) -> tuple[DocumentUnit, ...]:
    payload_tokens = await tokenizer.tokenize(
        document_prefix + unit.text, add_special=True
    )
    if len(payload_tokens) <= chunk_size_tokens:
        return (unit,)

    source_tokens = await tokenizer.tokenize(unit.text, add_special=False)
    children: list[DocumentUnit] = []
    start = 0
    while start < len(source_tokens):
        end, text = await _bounded_fitting_slice(
            source_tokens,
            start=start,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            max_source_tokens=max(1, chunk_size_tokens - prefix_token_count),
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
        children.append(child)

        if end == len(source_tokens):
            break
        next_start = end - chunk_overlap_tokens
        if next_start <= start:
            raise ValueError("token overlap leaves no room for new source tokens")
        start = next_start

    return tuple(children)


def _next_candidate_length(
    current_length: int,
    *,
    chunk_size_tokens: int,
    observed_payload_tokens: int,
) -> int:
    proportional = current_length * chunk_size_tokens // observed_payload_tokens
    geometric = current_length - max(1, (current_length + 7) // 8)
    return max(1, min(current_length - 1, proportional, geometric))


async def _bounded_fitting_slice(
    source_tokens: tuple[int, ...],
    start: int,
    *,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    max_source_tokens: int,
    document_prefix: str,
) -> tuple[int, str]:
    remaining_source_tokens = len(source_tokens) - start
    candidate_length = min(remaining_source_tokens, max_source_tokens)
    last_tested_length = 0

    for _ in range(_MAX_REFINEMENT_ATTEMPTS):
        last_tested_length = candidate_length
        end = start + candidate_length
        text = await tokenizer.detokenize(source_tokens[start:end])
        payload_tokens = await tokenizer.tokenize(
            document_prefix + text, add_special=True
        )
        if len(payload_tokens) <= chunk_size_tokens:
            return end, text
        if candidate_length == 1:
            return start, ""
        candidate_length = _next_candidate_length(
            candidate_length,
            chunk_size_tokens=chunk_size_tokens,
            observed_payload_tokens=len(payload_tokens),
        )

    if last_tested_length != 1:
        end = start + 1
        text = await tokenizer.detokenize(source_tokens[start:end])
        payload_tokens = await tokenizer.tokenize(
            document_prefix + text, add_special=True
        )
        if len(payload_tokens) <= chunk_size_tokens:
            return end, text

    return start, ""
