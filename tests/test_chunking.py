import json
from itertools import pairwise

import httpx
import pytest

from qemer_ingest.chunking import _bounded_fitting_slice, chunk_units
from qemer_ingest.models import DocumentUnit
from qemer_ingest.tokenization import TokenizationClient


class CharacterTokenizationClient:
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        special = (-1,) if add_special else ()
        return special + tuple(ord(character) for character in content)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in tokens)


class NonmonotonicTokenizationClient(CharacterTokenizationClient):
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if not add_special:
            return tuple(ord(character) for character in content)
        token_counts = {3: 5, 4: 5, 5: 4, 6: 6}
        return tuple(range(token_counts.get(len(content), len(content))))


class RecordingCharacterTokenizationClient(CharacterTokenizationClient):
    def __init__(self) -> None:
        self.detokenized_lengths: list[int] = []
        self.authoritative_contents: list[str] = []

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if add_special:
            self.authoritative_contents.append(content)
        return await super().tokenize(content, add_special=add_special)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        self.detokenized_lengths.append(len(tokens))
        return await super().detokenize(tokens)


class BoundaryInflatingTokenizationClient(RecordingCharacterTokenizationClient):
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if add_special:
            self.authoritative_contents.append(content)
            if len(content) == 7:
                return tuple(range(9))
            return (-1,) + tuple(ord(character) for character in content)
        return tuple(ord(character) for character in content)


class OneTokenFallbackTokenizationClient(CharacterTokenizationClient):
    def __init__(self) -> None:
        self.candidate_checks: list[int] = []

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        assert add_special
        self.candidate_checks.append(len(content))
        token_count = 1 if len(content) == 1 else 2_049
        return tuple(range(token_count))


def unit(kind: str = "prose", text: str = "short text") -> DocumentUnit:
    return DocumentUnit(
        "numpy-2.3-abc",
        kind,
        "Arrays",
        "https://example.test/README.md",
        text,
    )


@pytest.mark.asyncio
async def test_chunk_units_keeps_a_payload_at_exact_token_limit() -> None:
    original = unit(text="a" * 2047)
    chunks = await chunk_units(
        (original,),
        tokenizer=CharacterTokenizationClient(),
        chunk_size_tokens=2048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )
    assert chunks == (original,)


@pytest.mark.asyncio
async def test_chunk_units_splits_a_2049_token_payload_with_stable_metadata() -> None:
    chunks = await chunk_units(
        (unit(text="a" * 2048),),
        tokenizer=CharacterTokenizationClient(),
        chunk_size_tokens=2048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )
    assert [chunk.snippet_id for chunk in chunks] == [
        "numpy-2.3-abc-prose-001",
        "numpy-2.3-abc-prose-002",
    ]
    assert [chunk.text for chunk in chunks] == ["a" * 2047, "a"]
    assert [(chunk.kind, chunk.title, chunk.source_url) for chunk in chunks] == [
        ("prose", "Arrays", "https://example.test/README.md"),
        ("prose", "Arrays", "https://example.test/README.md"),
    ]


@pytest.mark.asyncio
async def test_chunk_units_accepts_a_deterministic_bounded_slice_for_nonmonotonic_counts() -> (
    None
):
    chunks = await chunk_units(
        (unit(text="abcdef"),),
        tokenizer=NonmonotonicTokenizationClient(),
        chunk_size_tokens=4,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert [chunk.text for chunk in chunks] == ["ab", "cd", "ef"]


@pytest.mark.asyncio
async def test_chunk_units_starts_a_large_unit_with_a_budget_sized_candidate() -> None:
    tokenizer = RecordingCharacterTokenizationClient()
    chunks = await chunk_units(
        (unit(text="a" * 20_680),),
        tokenizer=tokenizer,
        chunk_size_tokens=2_048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert tokenizer.detokenized_lengths[0] == 2_047
    assert max(tokenizer.detokenized_lengths) <= 2_047
    assert len(tokenizer.detokenized_lengths) == len(chunks)
    assert all(
        [
            len(await tokenizer.tokenize(chunk.text, add_special=True)) <= 2_048
            for chunk in chunks
        ]
    )


@pytest.mark.asyncio
async def test_chunk_units_refines_an_oversized_boundary_with_bounded_checks() -> None:
    tokenizer = BoundaryInflatingTokenizationClient()
    chunks = await chunk_units(
        (unit(text="abcdefghi"),),
        tokenizer=tokenizer,
        chunk_size_tokens=8,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert tokenizer.detokenized_lengths[:2] == [7, 6]
    assert [chunk.text for chunk in chunks] == ["abcdef", "ghi"]
    assert len(tokenizer.detokenized_lengths) <= len(chunks) * 9


@pytest.mark.asyncio
async def test_bounded_fitting_slice_checks_eight_candidates_plus_fallback() -> None:
    tokenizer = OneTokenFallbackTokenizationClient()
    source_tokens = tuple(ord("a") for _ in range(2_047))

    end, text = await _bounded_fitting_slice(
        source_tokens,
        0,
        tokenizer=tokenizer,
        chunk_size_tokens=2_048,
        max_source_tokens=2_047,
        document_prefix="",
    )

    assert (end, text) == (1, "a")
    assert all(length > 1 for length in tokenizer.candidate_checks[:8])
    assert tokenizer.candidate_checks[-1] == 1
    assert len(tokenizer.candidate_checks) == 9


@pytest.mark.asyncio
async def test_chunk_units_does_not_retokenize_an_accepted_child() -> None:
    tokenizer = RecordingCharacterTokenizationClient()
    chunks = await chunk_units(
        (unit(text="abcdefghi"),),
        tokenizer=tokenizer,
        chunk_size_tokens=7,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert [chunk.text for chunk in chunks] == ["abcdef", "ghi"]
    assert tokenizer.authoritative_contents.count("abcdef") == 1
    assert tokenizer.authoritative_contents.count("ghi") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "endpoint_message"),
    (("http", "/tokenize"), ("malformed", "tokenize response is invalid")),
)
async def test_chunk_units_adds_source_url_to_tokenize_errors(
    failure_kind: str, endpoint_message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure_kind == "http":
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"tokens": ["invalid"]})

    document = unit(text="abcdef")
    tokenizer = TokenizationClient(
        "http://embeddings.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ValueError) as error:
        async with tokenizer:
            await chunk_units(
                (document,),
                tokenizer=tokenizer,
                chunk_size_tokens=4,
                chunk_overlap_tokens=0,
                document_prefix="",
            )

    assert endpoint_message in str(error.value)
    assert document.source_url in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "endpoint_message"),
    (("http", "/detokenize"), ("malformed", "detokenize response is invalid")),
)
async def test_chunk_units_adds_source_url_to_detokenize_errors(
    failure_kind: str, endpoint_message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            payload = json.loads(request.content)
            content = payload["content"]
            add_special = payload["add_special"]
            special = [-1] if add_special else []
            return httpx.Response(
                200,
                json={"tokens": special + [ord(character) for character in content]},
            )
        if failure_kind == "http":
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"content": 1})

    document = unit(text="abc")
    tokenizer = TokenizationClient(
        "http://embeddings.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ValueError) as error:
        async with tokenizer:
            await chunk_units(
                (document,),
                tokenizer=tokenizer,
                chunk_size_tokens=3,
                chunk_overlap_tokens=0,
                document_prefix="",
            )

    assert endpoint_message in str(error.value)
    assert document.source_url in str(error.value)


@pytest.mark.asyncio
async def test_chunk_units_accounts_for_prefix_in_each_payload_budget() -> None:
    tokenizer = CharacterTokenizationClient()
    chunks = await chunk_units(
        (unit(text="abcdef"),),
        tokenizer=tokenizer,
        chunk_size_tokens=6,
        chunk_overlap_tokens=0,
        document_prefix="p:",
    )

    token_counts = [
        len(await tokenizer.tokenize("p:" + chunk.text, add_special=True))
        for chunk in chunks
    ]
    assert all(count <= 6 for count in token_counts)
    assert all("p:" not in chunk.text for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_units_repeats_exact_source_token_overlap() -> None:
    chunks = await chunk_units(
        (unit(text="abcdefgh"),),
        tokenizer=CharacterTokenizationClient(),
        chunk_size_tokens=6,
        chunk_overlap_tokens=2,
        document_prefix="",
    )

    assert all(
        current.text[:2] == previous.text[-2:] for previous, current in pairwise(chunks)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunk_size_tokens", "chunk_overlap_tokens", "message"),
    [
        (0, 0, "token size"),
        (-1, 0, "token size"),
        (1, -1, "token overlap"),
        (1, 1, "token overlap"),
    ],
)
async def test_chunk_units_rejects_invalid_token_options(
    chunk_size_tokens: int, chunk_overlap_tokens: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await chunk_units(
            (unit(),),
            tokenizer=CharacterTokenizationClient(),
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            document_prefix="",
        )


@pytest.mark.asyncio
async def test_chunk_units_rejects_a_prefix_that_leaves_no_source_room() -> None:
    with pytest.raises(
        ValueError, match="document prefix leaves no room for source tokens"
    ):
        await chunk_units(
            (unit(text="a"),),
            tokenizer=CharacterTokenizationClient(),
            chunk_size_tokens=3,
            chunk_overlap_tokens=0,
            document_prefix="ab",
        )
